/* SPDX-License-Identifier: MIT */
/* Shader capture controlled by PVR_DEBUG (pvr_g5p_shader_dump.h). */
#include "pvr_g5p_shader_dump.h"

#include <errno.h>
#include <stdarg.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "util/macros.h"
#include "util/log.h"
#include "pvr_debug.h"
#include "util/u_debug.h"
#include "c11/threads.h"

DEBUG_GET_ONCE_OPTION(pvr_shader_dump_dir, "PVR_SHADER_DUMP_DIR", "/tmp/pvr-shaders")
#include "util/mesa-blake3.h"
#include "util/u_atomic.h"
#include "util/u_call_once.h"
#include "vk_pipeline_layout.h"
#include "vk_shader_module.h"
#include "vk_util.h"
#include "pvr_descriptor_set.h"
#include "pvr_g5p_stub_spv.h"

static const char *dump_dir;
static bool stub_on;
static util_once_flag env_once = UTIL_ONCE_FLAG_INIT;

static void
read_env(void)
{
   const char *d = PVR_IS_DEBUG_SET(SHADER_DUMP) ? debug_get_option_pvr_shader_dump_dir() : NULL;
   if (d && *d) {
      mkdir(d, 0755); /* may exist */
      dump_dir = d;
      /* once per process: a sandboxed process may not write there (Firefox: MOZ_DISABLE_GPU_SANDBOX=1) */
      if (access(d, W_OK))
         mesa_logw("G5P shader dump: cannot write %s (pid %ld): %s", d, (long)getpid(), strerror(errno));
      else
         mesa_logi("G5P shader dump: %s (pid %ld)", d, (long)getpid());
   }
   stub_on = PVR_IS_DEBUG_SET(CAPTURE_STUB);
   if (stub_on)
      mesa_logw("G5P capture stub: ON (refused shaders are replaced; pictures will be wrong)");
}

const char *
pvr_g5p_shader_dump_dir(void)
{
   util_call_once(&env_once, read_env);
   return dump_dir;
}

bool
pvr_g5p_capture_stub(void)
{
   util_call_once(&env_once, read_env);
   return stub_on;
}

static bool
stage_spirv(const VkPipelineShaderStageCreateInfo *info, const uint32_t **words, size_t *bytes)
{
   VK_FROM_HANDLE(vk_shader_module, module, info->module);
   const VkShaderModuleCreateInfo *inline_ci = vk_find_struct_const(info->pNext, SHADER_MODULE_CREATE_INFO);
   if (module && !module->nir) {
      *words = (const uint32_t *)module->data;
      *bytes = module->size;
      return true;
   }
   if (inline_ci) {
      *words = inline_ci->pCode;
      *bytes = inline_ci->codeSize;
      return true;
   }
   return false; /* a driver-internal NIR module */
}

/* write @size bytes to @path through a temporary name and a rename */
static void
write_atomic(const char *path, const void *data, size_t size, bool keep_existing)
{
   static uint32_t serial;
   if (keep_existing && access(path, F_OK) == 0)
      return;
   char tmp[4096];
   snprintf(tmp, sizeof(tmp), "%s.tmp.%ld.%u", path, (long)getpid(), p_atomic_inc_return(&serial));
   int fd = open(tmp, O_WRONLY | O_CREAT | O_EXCL, 0644);
   if (fd < 0)
      return;
   const char *p = data;
   size_t left = size;
   while (left) {
      ssize_t n = write(fd, p, left);
      if (n <= 0) {
         if (n < 0 && errno == EINTR)
            continue;
         break;
      }
      p += n, left -= n;
   }
   close(fd);
   if (left || rename(tmp, path))
      unlink(tmp);
}

/* a growing string */
struct sbuf {
   char *s;
   size_t len, cap;
};

static void
sb_printf(struct sbuf *b, const char *fmt, ...) PRINTFLIKE(2, 3);

static void
sb_printf(struct sbuf *b, const char *fmt, ...)
{
   for (;;) {
      va_list ap;
      va_start(ap, fmt);
      int n = vsnprintf(b->s ? b->s + b->len : NULL, b->s ? b->cap - b->len : 0, fmt, ap);
      va_end(ap);
      if (n < 0)
         return;
      if (b->s && b->len + n < b->cap) {
         b->len += n;
         return;
      }
      size_t cap = MAX2(b->cap * 2, b->len + n + 256);
      char *s = realloc(b->s, cap);
      if (!s)
         return;
      b->s = s, b->cap = cap;
   }
}

static void
sb_json_string(struct sbuf *b, const char *s)
{
   sb_printf(b, "\"");
   for (; s && *s; s++) {
      const unsigned char c = *s;
      if (c == '"' || c == '\\')
         sb_printf(b, "\\%c", c);
      else if (c < 0x20)
         sb_printf(b, "\\u%04x", c);
      else
         sb_printf(b, "%c", c);
   }
   sb_printf(b, "\"");
}

static void
sb_hex(struct sbuf *b, const void *data, size_t size)
{
   sb_printf(b, "\"");
   for (size_t i = 0; i < size; i++)
      sb_printf(b, "%02x", ((const uint8_t *)data)[i]);
   sb_printf(b, "\"");
}

static const char *
descriptor_type_name(VkDescriptorType t)
{
   switch (t) {
   case VK_DESCRIPTOR_TYPE_SAMPLER: return "sampler";
   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: return "combined_image_sampler";
   case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: return "sampled_image";
   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: return "storage_image";
   case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER: return "uniform_texel_buffer";
   case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: return "storage_texel_buffer";
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: return "uniform_buffer";
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: return "storage_buffer";
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC: return "uniform_buffer_dynamic";
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: return "storage_buffer_dynamic";
   case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: return "input_attachment";
   default: return "other";
   }
}

void
pvr_g5p_shader_dump(const VkPipelineShaderStageCreateInfo *info, VkPipelineLayout layout_handle,
                    const char *stage, const struct pvr_g5p_dump_blob *blobs, unsigned nblobs,
                    const char *result, const char *reason)
{
   const char *dir = pvr_g5p_shader_dump_dir();
   const uint32_t *words;
   size_t bytes;
   if (!dir || !info || !stage_spirv(info, &words, &bytes))
      return;

   /* named by the first 20 bytes (40 hex digits) of the module's BLAKE3 */
   blake3_hash sha;
   char hex[BLAKE3_HEX_LEN];
   _mesa_blake3_compute(words, bytes, sha);
   _mesa_blake3_format(hex, sha);
   hex[40] = 0;
   char path[4096];
   snprintf(path, sizeof(path), "%s/%s.spv", dir, hex);
   write_atomic(path, words, bytes, true);

   /* everything that decides the compile, then the result */
   struct sbuf b = { 0 };
   sb_printf(&b, "{\n  \"spirv\": \"%s.spv\",\n  \"stage\": \"%s\",\n  \"entry\": ", hex, stage);
   sb_json_string(&b, info->pName);
   const VkSpecializationInfo *spec = info->pSpecializationInfo;
   sb_printf(&b, ",\n  \"specialization\": {\"entries\": [");
   for (uint32_t i = 0; spec && i < spec->mapEntryCount; i++)
      sb_printf(&b, "%s[%u, %u, %zu]", i ? ", " : "", spec->pMapEntries[i].constantID, spec->pMapEntries[i].offset,
                spec->pMapEntries[i].size);
   sb_printf(&b, "], \"data\": ");
   sb_hex(&b, spec ? spec->pData : NULL, spec ? spec->dataSize : 0);
   sb_printf(&b, "},\n  \"layout\": [");
   VK_FROM_HANDLE(vk_pipeline_layout, layout, layout_handle);
   for (uint32_t s = 0; layout && s < layout->set_count; s++) {
      const struct vk_descriptor_set_layout *vs = layout->set_layouts[s];
      sb_printf(&b, "%s\n    [", s ? "," : "");
      if (vs) {
         const struct pvr_descriptor_set_layout *ps = container_of(vs, struct pvr_descriptor_set_layout, vk);
         bool first = true;
         for (uint32_t i = 0; i < ps->binding_count; i++) {
            const struct pvr_descriptor_set_layout_binding *pb = &ps->bindings[i];
            if (!pb->descriptor_count)
               continue;
            sb_printf(&b, "%s{\"binding\": %u, \"type\": \"%s\", \"count\": %u, \"stages\": %u}", first ? "" : ", ",
                      i, descriptor_type_name(pb->type), pb->descriptor_count, pb->stage_flags);
            first = false;
         }
      }
      sb_printf(&b, "]");
   }
   sb_printf(&b, "],\n  \"push_constants\": [");
   for (uint32_t i = 0; layout && i < layout->push_range_count; i++)
      sb_printf(&b, "%s[%u, %u, %u]", i ? ", " : "", layout->push_ranges[i].stageFlags,
                layout->push_ranges[i].offset, layout->push_ranges[i].size);
   sb_printf(&b, "],\n  \"keys\": {");
   for (unsigned i = 0; i < nblobs; i++) {
      sb_printf(&b, "%s\"%s\": ", i ? ", " : "", blobs[i].name);
      sb_hex(&b, blobs[i].data, blobs[i].size);
   }
   sb_printf(&b, "}");
   if (!b.s)
      return;
   /* the variant: everything above */
   blake3_hash vsha;
   char vhex[BLAKE3_HEX_LEN];
   _mesa_blake3_compute(b.s, b.len, vsha);
   _mesa_blake3_format(vhex, vsha);
   vhex[12] = 0;
   sb_printf(&b, ",\n  \"result\": \"%s\",\n  \"reason\": ", result);
   sb_json_string(&b, reason ? reason : "");
   sb_printf(&b, "\n}\n");
   snprintf(path, sizeof(path), "%s/%s.%s.%s.json", dir, hex, stage, vhex);
   if (b.s)
      write_atomic(path, b.s, b.len, false);
   free(b.s);

   /* one index line, one append */
   char line[1024];
   int n = snprintf(line, sizeof(line), "%lld\t%ld\t%s\t%s\t%s\t%s\t", (long long)time(NULL), (long)getpid(),
                    hex, stage, vhex, result);
   for (const char *r = reason ? reason : ""; *r && n < (int)sizeof(line) - 2; r++)
      line[n++] = *r == '\t' || *r == '\n' ? ' ' : *r;
   line[MIN2(n, (int)sizeof(line) - 2)] = '\n';
   n = MIN2(n + 1, (int)sizeof(line) - 1);
   snprintf(path, sizeof(path), "%s/index.tsv", dir);
   int fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
   if (fd >= 0) {
      ssize_t w = write(fd, line, n);
      (void)w;
      close(fd);
   }
}

const VkPipelineShaderStageCreateInfo *
pvr_g5p_stub_stage(VkShaderStageFlagBits stage, VkPipelineShaderStageCreateInfo *info, VkShaderModuleCreateInfo *module)
{
   const uint32_t *code = stage == VK_SHADER_STAGE_FRAGMENT_BIT ? pvr_g5p_stub_frag :
                          stage == VK_SHADER_STAGE_VERTEX_BIT   ? pvr_g5p_stub_vert : pvr_g5p_stub_comp;
   const size_t size = stage == VK_SHADER_STAGE_FRAGMENT_BIT ? sizeof(pvr_g5p_stub_frag) :
                       stage == VK_SHADER_STAGE_VERTEX_BIT   ? sizeof(pvr_g5p_stub_vert) : sizeof(pvr_g5p_stub_comp);
   *module = (VkShaderModuleCreateInfo){ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                         .codeSize = size, .pCode = code };
   *info = (VkPipelineShaderStageCreateInfo){ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                              .pNext = module, .stage = stage, .pName = "main" };
   return info;
}
