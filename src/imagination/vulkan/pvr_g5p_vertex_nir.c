/* SPDX-License-Identifier: MIT */
/*
 * Vertex input of G5P (Apple A9) pipelines: the layout of the vertex
 * attributes in the unified store, and the vertex program's NIR rewritten to
 * read them there (pvr_g5p_vertex.h). Compiled only with
 * -Dimagination-g5p-experimental=true.
 *
 * The fetch is the A9 ICD's (g5p_pipeline.c vertex_input_ok() and
 * fetch_per_binding()): the vertex fetch PDS program DMAs raw dwords, one
 * DDMAD per fetch, and the vertex program reads location L from r4L..r4L+3.
 * That layout holds 8 locations in the backend's 32 input dwords, so past
 * location 7 the attributes are packed back to back instead, as Apple's
 * g3b-vtx16 program has them (our decode of the iOS capture: 16 attributes in
 * 19 dwords, 5 DDMADs over runs of adjacent dwords of the buffer), and the
 * program's loads are moved to the packed registers.
 *
 * 8- and 16-bit formats (normalized, scaled, f16) are fetched as the dwords
 * that hold their bytes, packed the same way, and the program converts each
 * component from its dword, as Apple's vertex programs convert (the
 * converted loads of pco_g5p_general.h, PCO_G5P_VS_CONVERT_BASE).
 */

#include <stdio.h>
#include <string.h>

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/shader_enums.h"
#include "util/macros.h"
#include "util/format/u_format.h"
#include "util/u_math.h"
#include "vk_format.h"
#include "vk_graphics_state.h"

#include "pvr_g5p_vertex.h"

#define REFUSE(l, ...) (snprintf((l)->refusal, sizeof((l)->refusal), __VA_ARGS__), false)

/* The ICD's layout: location L in r4L.. while every read location is below this */
#define ICD_LOCATIONS (PVR_G5P_VERTEX_INPUT_DWORDS / 4)

void
pvr_g5p_vertex_layout_init(struct pvr_g5p_vertex_layout *layout, const struct vk_vertex_input_state *vi,
                           bool dynamic_strides)
{
   memset(layout, 0, sizeof(*layout));
   layout->vi = vi;
   layout->dynamic_strides = dynamic_strides;
}

unsigned
pvr_g5p_vertex_format_dwords(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_R32_SFLOAT:
   case VK_FORMAT_R32_UINT:
   case VK_FORMAT_R32_SINT:
      return 1;
   case VK_FORMAT_R32G32_SFLOAT:
   case VK_FORMAT_R32G32_UINT:
   case VK_FORMAT_R32G32_SINT:
      return 2;
   case VK_FORMAT_R32G32B32_SFLOAT:
   case VK_FORMAT_R32G32B32_UINT:
   case VK_FORMAT_R32G32B32_SINT:
      return 3;
   case VK_FORMAT_R32G32B32A32_SFLOAT:
   case VK_FORMAT_R32G32B32A32_UINT:
   case VK_FORMAT_R32G32B32A32_SINT:
      return 4;
   default:
      return 0;
   }
}

bool
pvr_g5p_vertex_format_info(VkFormat format, struct pvr_g5p_vertex_format *f)
{
   const struct util_format_description *d = vk_format_description(format);
   memset(f, 0, sizeof(*f));
   if (!d || d->layout != UTIL_FORMAT_LAYOUT_PLAIN || d->block.width != 1 || d->block.height != 1 ||
       d->nr_channels < 1 || d->nr_channels > 4)
      return false;
   const struct util_format_channel_description *ch = &d->channel[0];
   const unsigned bits = ch->size;
   if ((bits != 8 && bits != 16 && bits != 32) || d->block.bits != bits * d->nr_channels)
      return false;
   for (unsigned i = 1; i < d->nr_channels; i++)
      if (d->channel[i].type != ch->type || d->channel[i].size != bits ||
          d->channel[i].normalized != ch->normalized || d->channel[i].pure_integer != ch->pure_integer)
         return false;
   f->components = d->nr_channels;
   f->bytes = bits / 8;
   for (unsigned c = 0; c < 4; c++) {
      const unsigned sw = d->swizzle[c];
      if (c < d->nr_channels && sw > PIPE_SWIZZLE_W)
         return false;
      f->swizzle[c] = sw <= PIPE_SWIZZLE_W ? sw : c;
   }
   /* every component of the format is read from its own channel */
   for (unsigned c = 0; c < d->nr_channels; c++)
      for (unsigned k = 0; k < c; k++)
         if (f->swizzle[c] == f->swizzle[k])
            return false;
   const bool is_signed = ch->type == UTIL_FORMAT_TYPE_SIGNED, is_unsigned = ch->type == UTIL_FORMAT_TYPE_UNSIGNED;
   if (bits == 32) {
      /* read as they are: float, or a pure integer */
      if (!(ch->type == UTIL_FORMAT_TYPE_FLOAT || ((is_signed || is_unsigned) && ch->pure_integer)))
         return false;
      f->convert = -1;
      return true;
   }
   if (ch->type == UTIL_FORMAT_TYPE_FLOAT) {
      if (bits != 16)
         return false;
      f->convert = PCO_G5P_VSF_F16;
      return true;
   }
   /* to float: normalized or scaled; 8- and 16-bit integer inputs are not
    * converted yet */
   if (!(is_signed || is_unsigned) || ch->pure_integer)
      return false;
   f->convert = (bits == 16 ? PCO_G5P_VSF_U16 : PCO_G5P_VSF_U8) + (ch->normalized ? 2 : 0) + is_signed;
   return true;
}

struct attrib {
   unsigned location, binding, dwords;
   uint32_t offset; /* of the first fetched dword */
   uint32_t lead;   /* the attribute's byte in that dword */
   struct pvr_g5p_vertex_format f;
};

/* A vertex attribute load the layout handles: a direct generic location */
static int
generic_location(const nir_intrinsic_instr *in)
{
   if (in->intrinsic != nir_intrinsic_load_input)
      return -1;
   const int loc = (int)nir_intrinsic_io_semantics(in).location - VERT_ATTRIB_GENERIC0;
   return loc >= 0 && loc < PCO_G5P_MAX_ATTRIBS ? loc : -1;
}

/* A binding's fetch index: per vertex, per instance (divisor 1), the first
 * instance's element (divisor 0), or the instance divided (divisor > 1) */
static uint8_t
fetch_rate(const struct vk_vertex_input_state *vi, unsigned binding)
{
   if (vi->bindings[binding].input_rate == VK_VERTEX_INPUT_RATE_VERTEX)
      return PVR_G5P_FETCH_VERTEX;
   const uint32_t d = vi->bindings[binding].divisor;
   return d == 1 ? PVR_G5P_FETCH_INSTANCE : d ? PVR_G5P_FETCH_INSTANCE_DIVISOR : PVR_G5P_FETCH_INSTANCE_DIVISOR0;
}

/* The fetches' divisors; at most PVR_G5P_MAX_DIVISORS distinct ones > 1 */
static bool
fetch_divisors(struct pvr_g5p_vertex_layout *l)
{
   uint32_t seen[PVR_G5P_MAX_DIVISORS];
   unsigned n = 0;
   for (unsigned f = 0; f < l->fetch_count; f++) {
      struct pvr_g5p_fetch *fe = &l->fetch_list[f];
      fe->divisor = l->vi->bindings[fe->binding].divisor;
      if (fe->rate != PVR_G5P_FETCH_INSTANCE_DIVISOR)
         continue;
      unsigned k = 0;
      while (k < n && seen[k] != fe->divisor)
         k++;
      if (k == n) {
         if (n == PVR_G5P_MAX_DIVISORS)
            return REFUSE(l, "more than %u distinct vertex attribute divisors", PVR_G5P_MAX_DIVISORS);
         seen[n++] = fe->divisor;
      }
   }
   return true;
}

/* The ICD's fetches: each binding whose attributes are consecutive whole
 * vec4s is one DDMAD into r4L.. of its first location; otherwise one DDMAD
 * per attribute, format-sized, into r4L */
static void
icd_fetches(struct pvr_g5p_vertex_layout *l, const struct attrib *a, unsigned n)
{
   const struct vk_vertex_input_state *vi = l->vi;
   bool per_binding = true;
   for (unsigned i = 0; i < n && per_binding; i++)
      per_binding = vi->attributes[a[i].location].format == VK_FORMAT_R32G32B32A32_SFLOAT;
   for (unsigned i = 0; i < n && per_binding; i++) {
      unsigned first = ~0u, cnt = 0;
      uint32_t min_off = ~0u;
      for (unsigned k = 0; k < n; k++) {
         if (a[k].binding != a[i].binding)
            continue;
         first = MIN2(first, a[k].location);
         min_off = MIN2(min_off, a[k].offset);
         cnt++;
      }
      if (a[i].offset != min_off + 16 * (a[i].location - first) ||
          (!l->dynamic_strides && vi->bindings[a[i].binding].stride < 16 * cnt))
         per_binding = false;
   }
   l->fetch_count = 0;
   if (per_binding) {
      for (unsigned b = 0; b < MESA_VK_MAX_VERTEX_BINDINGS; b++) {
         unsigned first = ~0u, cnt = 0;
         uint32_t min_off = ~0u;
         for (unsigned i = 0; i < n; i++) {
            if (a[i].binding != b)
               continue;
            first = MIN2(first, a[i].location);
            min_off = MIN2(min_off, a[i].offset);
            cnt++;
         }
         if (cnt)
            l->fetch_list[l->fetch_count++] = (struct pvr_g5p_fetch){
               .binding = b, .offset = min_off, .dwords = 4 * cnt, .ao = 4 * first, .rate = fetch_rate(vi, b) };
      }
   } else {
      for (unsigned i = 0; i < n; i++)
         l->fetch_list[l->fetch_count++] =
            (struct pvr_g5p_fetch){ .binding = a[i].binding, .offset = a[i].offset, .dwords = a[i].dwords,
                                    .ao = 4 * a[i].location, .rate = fetch_rate(vi, a[i].binding) };
   }
   for (unsigned i = 0; i < n; i++)
      l->components[a[i].location] = a[i].dwords;
}

static int
attrib_order(const void *x, const void *y)
{
   const struct attrib *a = x, *b = y;
   if (a->binding != b->binding)
      return a->binding < b->binding ? -1 : 1;
   if (a->offset != b->offset)
      return a->offset < b->offset ? -1 : 1;
   return a->location < b->location ? -1 : a->location > b->location;
}

/* Packed: the attributes in buffer order, back to back from r0; a run of
 * attributes that are adjacent in one binding is one DDMAD (Apple's
 * g3b-vtx16 form). base[L] is location L's first register. */
static bool
packed_fetches(struct pvr_g5p_vertex_layout *l, struct attrib *a, unsigned n, unsigned base[PCO_G5P_MAX_ATTRIBS])
{
   qsort(a, n, sizeof(*a), attrib_order);
   unsigned slot = 0;
   l->fetch_count = 0;
   for (unsigned i = 0; i < n; i++) {
      struct pvr_g5p_fetch *last = l->fetch_count ? &l->fetch_list[l->fetch_count - 1] : NULL;
      base[a[i].location] = slot;
      if (last && last->binding == a[i].binding && last->offset + 4 * last->dwords == a[i].offset) {
         last->dwords += a[i].dwords;
      } else {
         l->fetch_list[l->fetch_count++] =
            (struct pvr_g5p_fetch){ .binding = a[i].binding, .offset = a[i].offset, .dwords = a[i].dwords,
                                    .ao = slot, .rate = fetch_rate(l->vi, a[i].binding) };
      }
      slot += a[i].dwords;
   }
   if (slot > PVR_G5P_VERTEX_INPUT_DWORDS)
      return REFUSE(l, "vertex attributes of %u dwords (the unified store holds %u)", slot,
                    PVR_G5P_VERTEX_INPUT_DWORDS);
   for (unsigned v = 0; 4 * v < slot; v++)
      l->components[v] = MIN2(4, slot - 4 * v);
   return true;
}

struct rewrite {
   const struct pvr_g5p_vertex_layout *l;
   const unsigned *base;
};

/* A load of location L, components first.., from the packed registers:
 * one dword each, at virtual location reg / 4, component reg % 4; the
 * components past the format are (0, 0, 0, 1) */
static bool
rewrite_load(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
   const struct rewrite *rw = data;
   const int loc = generic_location(in);
   if (loc < 0)
      return false;
   struct pvr_g5p_vertex_format f;
   pvr_g5p_vertex_format_info(rw->l->vi->attributes[loc].format, &f);
   const unsigned have = f.components, lead = rw->l->vi->attributes[loc].offset % 4;
   const nir_alu_type type = nir_intrinsic_dest_type(in);
   const bool is_float = nir_alu_type_get_base_type(type) == nir_type_float;
   b->cursor = nir_before_instr(&in->instr);
   nir_def *c[4];
   for (unsigned k = 0; k < in->def.num_components; k++) {
      const unsigned comp = nir_intrinsic_component(in) + k;
      if (comp >= have) {
         c[k] = comp != 3 ? nir_imm_int(b, 0) : is_float ? nir_imm_float(b, 1.0f) : nir_imm_int(b, 1);
         continue;
      }
      /* the component's byte in the fetched dwords; a converted one names
       * its format and byte in the load's base */
      const unsigned at = lead + f.bytes * f.swizzle[comp], reg = rw->base[loc] + at / 4;
      const unsigned base = f.convert < 0 ? reg / 4 : PCO_G5P_VS_CONVERT_BASE + 4 * f.convert + at % 4;
      c[k] = nir_load_input(b, 1, 32, nir_imm_int(b, 0), .base = base, .component = reg % 4, .dest_type = type,
                            .io_semantics = (nir_io_semantics){ .location = VERT_ATTRIB_GENERIC0 + reg / 4,
                                                                .num_slots = 1 });
   }
   nir_def_replace(&in->def, nir_vec(b, c, in->def.num_components));
   return true;
}

bool
pvr_g5p_vertex_lower(nir_shader *vs, struct pvr_g5p_vertex_layout *l)
{
   const struct vk_vertex_input_state *vi = l->vi;

   /* the locations the program reads */
   l->read = 0;
   nir_foreach_function_impl (impl, vs) {
      nir_foreach_block (block, impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
            if (in->intrinsic != nir_intrinsic_load_input)
               continue;
            const int loc = generic_location(in);
            if (loc < 0)
               return REFUSE(l, "a vertex input at location %d", (int)nir_intrinsic_io_semantics(in).location);
            if (in->def.bit_size != 32 || !nir_src_is_const(in->src[0]) || nir_src_as_uint(in->src[0]))
               return REFUSE(l, "vertex input %d: an indirect or %u-bit load", loc, in->def.bit_size);
            l->read |= BITFIELD_BIT(loc);
         }
      }
   }

   const uint32_t valid = vi ? vi->attributes_valid : 0;
   if (l->read & ~valid)
      return REFUSE(l, "the vertex shader reads location %u, which has no vertex attribute",
                    u_bit_scan(&(uint32_t){ l->read & ~valid }));

   struct attrib a[PCO_G5P_MAX_ATTRIBS];
   unsigned n = 0;
   bool converted = false;
   u_foreach_bit (loc, valid) {
      const struct vk_vertex_attribute_state *at = &vi->attributes[loc];
      struct pvr_g5p_vertex_format f;
      if (!pvr_g5p_vertex_format_info(at->format, &f))
         return REFUSE(l, "vertex attribute %u: format %d (32-bit float and integer, 8/16-bit normalized and scaled, "
                       "16-bit float formats only)", loc, at->format);
      if (!(l->read & BITFIELD_BIT(loc)))
         continue; /* bound but not read: not fetched */
      if (loc >= PCO_G5P_MAX_ATTRIBS)
         return REFUSE(l, "vertex attribute at location %u", loc);
      /* the fetch is dword-granular: a 32-bit attribute starts on a dword,
       * a converted one is read from the dwords that hold its bytes */
      const unsigned lead = at->offset % 4;
      if (f.convert < 0 && lead)
         return REFUSE(l, "vertex attribute %u at offset %u (the fetch is dword-granular)", loc, at->offset);
      if (lead % f.bytes)
         return REFUSE(l, "vertex attribute %u: %u-byte components at offset %u", loc, f.bytes, at->offset);
      if (!l->dynamic_strides && vi->bindings[at->binding].stride % 4)
         return REFUSE(l, "vertex binding %u: stride %u (the fetch is dword-granular)", at->binding,
                       vi->bindings[at->binding].stride);
      converted |= f.convert >= 0;
      a[n++] = (struct attrib){ loc, at->binding, DIV_ROUND_UP(lead + f.bytes * f.components, 4), at->offset - lead,
                                lead, f };
   }

   memset(l->components, 0, sizeof(l->components));
   /* the ICD's layout holds 32-bit attributes below location 8 */
   l->packed = converted || l->read >= BITFIELD_BIT(ICD_LOCATIONS);
   if (!l->packed) {
      icd_fetches(l, a, n);
      return fetch_divisors(l);
   }

   unsigned base[PCO_G5P_MAX_ATTRIBS] = { 0 };
   if (!packed_fetches(l, a, n, base) || !fetch_divisors(l))
      return false;
   const struct rewrite rw = { l, base };
   nir_shader_intrinsics_pass(vs, rewrite_load, nir_metadata_control_flow, (void *)&rw);
   nir_opt_constant_folding(vs);
   nir_opt_dce(vs);
   vs->info.inputs_read = 0;
   for (unsigned v = 0; v < PCO_G5P_MAX_ATTRIBS && l->components[v]; v++)
      vs->info.inputs_read |= BITFIELD64_BIT(VERT_ATTRIB_GENERIC0 + v);
   nir_index_ssa_defs(nir_shader_get_entrypoint(vs));
   return true;
}
