/* SPDX-License-Identifier: MIT
 * Generated-by: LLM
 * Ordinary linear A9 descriptors, with the logical element count kept in
 * software-only descriptor words6/7. Field encoders are shared with images.
 */
#include "pvr_g5p_texel_buffer.h"
#include "pvr_g5p_tex_encode.h"
#include "pvr_g5p_formats.h"
#include "pvr_descriptor_set.h"
#include "pvr_device.h"
#include "pvr_tex_state.h"
#include "pco_g5p_general.h"
#include "vk_format.h"
#include "vk_log.h"
#include "vk_enum_to_str.h"
#include "util/log.h"
VkResult pvr_g5p_tbo_pack(struct pvr_device *device, const struct pvr_texture_state_info *info,
                         struct pvr_image_descriptor *state)
{
   uint32_t key;
   bool srgb;
   const char *why = NULL;
   unsigned cpp = vk_format_get_blocksize(info->format);
   if (!pvr_g5p_tbo_probe()) why = "texel buffers require hardware qualification (PVR_G5P_TEXEL_BUFFER=probe)";
   else if (!pvr_g5p_tbo_format(info->format) || !pvr_g5p_format_tex(info->format, &key, &srgb)) why = "unqualified texel-buffer format";
   else if (info->addr.addr % PVR_G5P_TBO_ALIGN) why = "texel buffer base is not 64-byte aligned";
   else if (!info->buffer_elems || info->buffer_elems > PVR_G5P_TBO_LIMIT) why = "texel-buffer range outside 1..65536 elements";
   if (why) {
      mesa_loge("G5P refuses: %s: %s", vk_Format_to_str(info->format), why);
      return vk_errorf(device, VK_ERROR_FORMAT_NOT_SUPPORTED, "G5P refuses: %s: %s", vk_Format_to_str(info->format), why);
   }
   unsigned rows = (info->buffer_elems + PVR_G5P_TBO_WIDTH - 1) / PVR_G5P_TBO_WIDTH;
   uint64_t q[3];
   pvr_g5p_tex_linear(q, key, false, info->addr.addr, PVR_G5P_TBO_WIDTH, rows, PVR_G5P_TBO_WIDTH * cpp);
   memset(state, 0, sizeof(*state));
   state->words[0] = q[0];state->words[1] = q[1];
   state->meta[0] = q[2];state->meta[1] = q[2] >> 32;
   state->meta[2] = info->buffer_elems;state->meta[3] = PVR_G5P_TBO_MAGIC;
   return VK_SUCCESS;
}
bool pvr_g5p_tbo_upload(unsigned kind, unsigned level, unsigned dwords, const uint32_t desc[8], uint32_t out[8])
{
   if (desc[7] != PVR_G5P_TBO_MAGIC || !desc[6]) return false;
   memset(out, 0, 8 * sizeof(*out));
   switch (kind) {
   case PCO_G5P_UPLOAD_IMAGE:
      if (dwords != 8) return false;
      /* The final qword belongs to software range metadata. Hardware sees
       * only the generated image state and a zero reserved qword. */
      memcpy(out, desc, 6 * sizeof(*out));return true;
   case PCO_G5P_UPLOAD_IMAGE_SIZE:
      if (dwords != 4 || level) return false;
      out[0] = desc[6];out[1] = 0; /* reserved, explicit common-store level0 */
      out[2] = out[3] = 1;return true;
   case PCO_G5P_UPLOAD_SAMPLER: {
      if (dwords != 2) return false;
      /* Native read ignores the sampler, but the existing DMA ABI reserves
       * its selector. Supply the normal nearest/clamped single-level state. */
      uint64_t word = PVR_G5P_SMP_BASE;
      out[0] = word;out[1] = word >> 32;return true;
   }
   default: return false;
   }
}
