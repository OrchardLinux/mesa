/* SPDX-License-Identifier: MIT */
#include "../g5p/g5p_usc.h"
#include "util/half_float.h"
/*
 * A9 compute samples (WGSL textureSampleLevel/Grad/Load/Dimensions in a
 * compute shader). Apple's A9 kernels (sample(s, uv, level(l)),
 * gradient2d(), read(g, lod)) use the
 * coupled operand tuples of the qualified vertex sample (g5p_usc_sample_ext()
 * G5P_TEX_TUPLE_COUPLED_*), byte for byte:
 *   level(l)  G5P_TEX_TUPLE_COUPLED_TEMP_LEVEL, 2D, four channels, f16 level in r2
 *   gradient  G5P_TEX_TUPLE_COUPLED_GRAD, 2D, four channels, [dx.x dy.x dx.y dy.y] f32
 * The f16 level is Apple's f32-to-f16 conversion (g9p_f16_of(), the vertex
 * level form without a last-use flag). Every sample here is followed by a
 * wait, so each one issues in the state of Apple's single-sample kernels.
 *
 * Descriptor words go to fixed common-store dwords, inside the 64 the
 * dispatch's DMA task already loads (pvr_arch_g5p_compute.c): texture
 * states at 32 and 48 (four-qword aligned), sampler words at 58, 60, 62,
 * size blocks (textureDimensions, CPU-computed as the fragment path's) at 40
 * and 44 when no storage image uses 40-47. Buffers keep dwords 0-2n, the
 * device fence 56-57, a storage image 40-47 and its level 31.
 */
static const uint8_t a9_tex_image_dw[] = { 32, 48 };
static const uint8_t a9_tex_sampler_dw[] = { 58, 60, 62 };
static const uint8_t a9_tex_size_dw[] = { 40, 44 };

static void
a9_tex_refuse(struct g9p_ctx *ctx, const char *why)
{
   g9p_fail(ctx->out, "%s", why);
}

/* The common dword of an upload, added if new; -1 when its slots are full */
static int
a9_tex_upload(struct g9p_ctx *ctx, unsigned kind, unsigned set, unsigned binding, unsigned element,
              unsigned combined, unsigned level)
{
   struct g9p_shader *out = ctx->out;
   const uint8_t *slots = kind == G9P_A9_UP_IMAGE ? a9_tex_image_dw :
                          kind == G9P_A9_UP_SIZE ? a9_tex_size_dw : a9_tex_sampler_dw;
   const unsigned nslots = kind == G9P_A9_UP_IMAGE ? ARRAY_SIZE(a9_tex_image_dw) :
                           kind == G9P_A9_UP_SIZE ? ARRAY_SIZE(a9_tex_size_dw) : ARRAY_SIZE(a9_tex_sampler_dw);
   unsigned used = 0;

   for (unsigned i = 0; i < out->a9_num_uploads; i++) {
      const struct g9p_a9_upload *u = &out->a9_uploads[i];
      const bool same_class = u->kind == kind ||
         ((u->kind == G9P_A9_UP_SAMPLER || u->kind == G9P_A9_UP_FETCH_SAMPLER) &&
          (kind == G9P_A9_UP_SAMPLER || kind == G9P_A9_UP_FETCH_SAMPLER));
      if (u->kind == kind && u->set == set && u->binding == binding && u->element == element &&
          u->combined == combined && u->level == level)
         return u->common;
      used += same_class;
   }
   if (used == nslots || out->a9_num_uploads == G9P_A9_MAX_UPLOADS || set > 255 || binding > 255 ||
       element > 255 || level > 255) {
      a9_tex_refuse(ctx, kind == G9P_A9_UP_IMAGE ? "more than 2 sampled textures in a compute shader" :
                         kind == G9P_A9_UP_SIZE ? "more than 2 texture size queries in a compute shader" :
                         "more than 3 samplers in a compute shader");
      return -1;
   }
   out->a9_uploads[out->a9_num_uploads++] = (struct g9p_a9_upload){
      .kind = kind, .set = set, .binding = binding, .element = element, .combined = combined,
      .level = level, .common = slots[used],
      .dwords = kind == G9P_A9_UP_IMAGE ? 8 : kind == G9P_A9_UP_SIZE ? 4 : 2,
   };
   return slots[used];
}

/* ra_f/sla_f/slc_f use common-store integer/f16 levels, not the
 * temporary continuation of the older 2D samples. Three literal slots
 * precede the storage-image level at 31; buffer pointers must end at 27. */
static int
a9_tex_shared_lod(struct g9p_ctx *ctx, nir_tex_instr *tex, bool read)
{
   int li = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   assert(li < 0 || nir_src_is_const(tex->src[li].src));
   uint32_t value = li < 0 ? 0 : read ? nir_src_as_uint(tex->src[li].src) :
                                       _mesa_float_to_half(nir_src_as_float(tex->src[li].src));
   unsigned used = 0;
   for (unsigned i = 0; i < ctx->out->a9_num_uploads; i++) {
      const struct g9p_a9_upload *u = &ctx->out->a9_uploads[i];
      if (u->kind != G9P_A9_UP_LOD) continue;
      if (u->value == value) return u->common;
      used++;
   }
   if (used == 3 || ctx->out->a9_num_uploads == G9P_A9_MAX_UPLOADS) {
      a9_tex_refuse(ctx, "more than 3 shared texture levels in a compute shader");
      return -1;
   }
   ctx->out->a9_uploads[ctx->out->a9_num_uploads++] = (struct g9p_a9_upload) {
      .kind = G9P_A9_UP_LOD, .common = 28 + used, .dwords = 1, .value = value,
   };
   return 28 + used;
}

static bool
a9_tex_var(nir_tex_instr *tex, nir_tex_src_type type, unsigned *set, unsigned *binding, unsigned *element)
{
   const int i = nir_tex_instr_src_index(tex, type);
   if (i < 0)
      return false;
   nir_deref_instr *deref = nir_src_as_deref(tex->src[i].src);
   nir_variable *var = nir_deref_instr_get_variable(deref);
   if (!var)
      return false;
   *set = var->data.descriptor_set;
   *binding = var->data.binding;
   *element = 0;
   if (deref->deref_type == nir_deref_type_array) {
      if (!nir_src_is_const(deref->arr.index))
         return false;
      *element = nir_src_as_uint(deref->arr.index);
   } else if (deref->deref_type != nir_deref_type_var) {
      return false;
   }
   return true;
}

static void
a9_tex_emit_raw(struct g9p_ctx *ctx, const uint8_t *raw, unsigned n)
{
   for (unsigned i = 0; i + 1 < n; i += 2)
      G9P_EMIT(ctx->c, raw[i] | raw[i + 1] << 8);
}

/* A value copied into register @dst (OR copies any bit pattern) */
static bool
a9_tex_copy(struct g9p_ctx *ctx, unsigned dst, nir_def *def, unsigned comp)
{
   const int r = src_reg(ctx, nir_get_scalar(def, comp));
   if (r < 0)
      return false;
   g9p_alu2(ctx->c, G9P_OR, dst, r, r, false);
   return true;
}

/* textureDimensions / textureNumLevels at a constant level: words of the
 * CPU-computed size block (pvr_g5p_tex_size(): w, h, depth/layers, levels) */
static bool
a9_txs(struct g9p_ctx *ctx, nir_tex_instr *tex)
{
   unsigned set, binding, element, level = 0;
   const int li = nir_tex_instr_src_index(tex, nir_tex_src_lod);

   if (!a9_tex_var(tex, nir_tex_src_texture_deref, &set, &binding, &element)) {
      a9_tex_refuse(ctx, "texture size query of a texture without a constant binding");
      return false;
   }
   if (tex->op == nir_texop_txs && li >= 0) {
      if (!nir_src_is_const(tex->src[li].src)) {
         a9_tex_refuse(ctx, "textureDimensions at a non-constant level in a compute shader");
         return false;
      }
      level = nir_src_as_uint(tex->src[li].src);
   }
   const int common = a9_tex_upload(ctx, G9P_A9_UP_SIZE, set, binding, element, 0, level);
   if (common < 0)
      return false;
   int zero = const_reg(ctx, 0);
   if (zero < 0 || (zero = low_reg(ctx, zero)) < 0)
      return false;
   const unsigned n = tex->op == nir_texop_query_levels ? 1 : tex->def.num_components;
   int dst[4];
   for (unsigned c = 0; c < n; c++) {
      if (!ctx->last_use[key(&tex->def, c)]) {
         dst[c] = -1;
         continue;
      }
      dst[c] = alloc_reg(ctx);
      if (dst[c] < 0)
         return false;
      g9p_mov_uniform(ctx->c, dst[c], common + (tex->op == nir_texop_query_levels ? 3 : c), zero);
   }
   release(ctx, &tex->instr);
   for (unsigned c = 0; c < n; c++)
      ctx->reg[key(&tex->def, c)] = dst[c];
   return true;
}

static bool
a9_tex(struct g9p_ctx *ctx, nir_tex_instr *tex)
{
   if (tex->op == nir_texop_txs || tex->op == nir_texop_query_levels)
      return a9_txs(ctx, tex);

   const bool read = tex->op == nir_texop_txf, grad = tex->op == nir_texop_txd;
   const int ci = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   unsigned set, binding, element, sset, sbinding, selement, combined = 0;

   if (tex->op == nir_texop_tex || tex->op == nir_texop_txb) {
      a9_tex_refuse(ctx, "an implicit-LOD or bias sample in a compute shader (no derivatives)");
      return false;
   }
   if (tex->op != nir_texop_txl && !grad && !read) {
      a9_tex_refuse(ctx, "texture operation not supported in compute shaders yet");
      return false;
   }
   const bool array = tex->sampler_dim == GLSL_SAMPLER_DIM_2D && tex->is_array;
   const bool cube = tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE;
   const bool layered = array || cube;
   if (tex->is_shadow ||
       (tex->sampler_dim != GLSL_SAMPLER_DIM_2D && !cube) ||
       (cube && (tex->is_array || tex->op != nir_texop_txl)) || (array && grad)) {
      char why[160];
      const char *dim = cube ? "cube" : tex->sampler_dim == GLSL_SAMPLER_DIM_3D ? "3D" :
                        array ? "2D array" : tex->sampler_dim == GLSL_SAMPLER_DIM_2D ? "2D" : "other";
      snprintf(why, sizeof(why), "compute %s on %s is_array=%u is_shadow=%u unsupported",
               (read ? "txf" : grad ? "txd" : "txl"), dim, tex->is_array, tex->is_shadow);
      a9_tex_refuse(ctx, why);
      return false;
   }
   if (nir_alu_type_get_base_type(tex->dest_type) != nir_type_float) {
      a9_tex_refuse(ctx, "an integer-texture sample or load in a compute shader");
      return false;
   }
   for (unsigned i = 0; i < tex->num_srcs; i++) {
      switch (tex->src[i].src_type) {
      case nir_tex_src_coord: case nir_tex_src_lod: case nir_tex_src_ddx: case nir_tex_src_ddy:
      case nir_tex_src_texture_deref: case nir_tex_src_sampler_deref:
         break;
      default:
         a9_tex_refuse(ctx, "a texel offset or other extra texture source in a compute shader");
         return false;
      }
   }
   if (ci < 0 || !a9_tex_var(tex, nir_tex_src_texture_deref, &set, &binding, &element)) {
      a9_tex_refuse(ctx, "a compute sample without a constant texture binding");
      return false;
   }
   if (read) {
      sset = sbinding = selement = 0;
   } else if (!a9_tex_var(tex, nir_tex_src_sampler_deref, &sset, &sbinding, &selement)) {
      /* a combined image sampler: its sampler words are at +32 of the same descriptor */
      sset = set, sbinding = binding, selement = element, combined = 1;
   }
   const int image = a9_tex_upload(ctx, G9P_A9_UP_IMAGE, set, binding, element, 0, 0);
   const int sampler = image < 0 ? -1 :
      read ? a9_tex_upload(ctx, G9P_A9_UP_FETCH_SAMPLER, 0, 0, 0, 0, 0) :
             a9_tex_upload(ctx, G9P_A9_UP_SAMPLER, sset, sbinding, selement, combined, 0);
   if (sampler < 0)
      return false;

   /* coordinates: a fresh block, consumed by the sample */
   const unsigned coords = layered ? 3 : 2;
   const int block = alloc_range(ctx, coords);
   if (block < 0 || block + coords > 32)
      return false;
   ctx->temps |= BITFIELD64_RANGE(block, coords);
   for (unsigned c = 0; c < coords; c++)
      if (!a9_tex_copy(ctx, block + c, tex->src[ci].src.ssa, c))
         return false;

   /* level (f16 for samples, u32 for reads) or gradients */
   int lod;
   const int li = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   const bool register_lod = layered && li >= 0 && !nir_src_is_const(tex->src[li].src);
   if (register_lod) {
      /* Apple's A9 witnesses sla_d/slc_d convert the level to the low f16 of
       * a register with the source in slot A, its last read marked; the
       * older plain-2D g9p_f16_of() reads slot B; leave that proven path
       * alone. Copy into a private low temporary: the witness marks the
       * source's last read, so it cannot be a still-live NIR value. */
      int src = src_reg(ctx, nir_get_scalar(tex->src[li].src.ssa, 0));
      if (src < 0) return false;
      int copy = alloc_low_temp(ctx);
      if (copy < 0) return false;
      g9p_alu2(ctx->c, G9P_OR, copy, src, src, false);
      lod = read ? copy : alloc_low_temp(ctx);
      if (lod < 0) return false;
      if (!read) {
         const unsigned at = ctx->c->n;
         g9p_f16_of(ctx->c, lod, 0);
         (*g9p_code_word(ctx->c, at + 2)) |= copy << G9P_SRC_A_SHIFT | G9P_SRC_A_LAST_READ;
      }
      /* ra_d loads its uint LOD directly into r0: no f16 conversion. */
   } else if (layered) {
      lod = a9_tex_shared_lod(ctx, tex, read);
      if (lod < 0) return false;
   } else if (grad) {
      const int dxi = nir_tex_instr_src_index(tex, nir_tex_src_ddx);
      const int dyi = nir_tex_instr_src_index(tex, nir_tex_src_ddy);
      lod = alloc_range(ctx, 4);
      if (dxi < 0 || dyi < 0 || lod < 0 || lod > 28)
         return false;
      ctx->temps |= BITFIELD64_RANGE(lod, 4);
      for (unsigned c = 0; c < 2; c++) {
         if (!a9_tex_copy(ctx, lod + 2 * c, tex->src[dxi].src.ssa, c) ||
             !a9_tex_copy(ctx, lod + 2 * c + 1, tex->src[dyi].src.ssa, c))
            return false;
      }
   } else {
      const int li = nir_tex_instr_src_index(tex, nir_tex_src_lod);
      int r = li >= 0 ? src_reg(ctx, nir_get_scalar(tex->src[li].src.ssa, 0)) : const_reg(ctx, 0);
      if (r < 0)
         return false;
      if (read) {
         lod = alloc_temp(ctx);
         if (lod < 0 || lod > 31)
            return false;
         g9p_alu2(ctx->c, G9P_OR, lod, r, r, false);
      } else {
         r = one_src(ctx, r);
         lod = r < 0 ? r : alloc_low_temp(ctx);
         if (lod < 0)
            return false;
         g9p_f16_of(ctx->c, lod, r);
      }
   }
   if (cube) {
      /* slc_f's cube coordinates (g9p_cube_coords()): bank0 r0..2 directions -> bank1 coords.
       * Bank0 homes start at r32, so this scratch cannot overwrite them. */
      for (unsigned c = 0; c < 3; c++)
         g9p_mov_to_high(ctx->c, c, block + c);
      g9p_cube_coords(ctx->c, 2 * block, 0);
   }
   release(ctx, &tex->instr);
   const int dst = alloc_range(ctx, 4);
   if (dst < 0 || dst > 28)
      return false;

   if (register_lod) {
      /* Apple's A9 register-LOD array or cube sample (slc_d/sla_d) and fetch
       * (ra_d, coordinate r3; its preceding ops produce x,y,layer in r3,r4,r5):
       * g5p_usc_sample_register_lod(), the coupled temporary-level tuple
       * with the LOD register. Keep this local to these witnessed compute
       * cases; g5p_usc_sample_ext() and all graphics encodings are unchanged. */
      const struct g5p_tex_ext x = {
         .dst_half = 2 * dst, .image_qword = image / 2, .sampler_qword = sampler / 2, .coord_dword = block,
         .dim = cube ? G5P_TEX_CUBE : G5P_TEX_2D_ARRAY, .mask = 0xf, .read = read,
         .lod = G5P_TEX_LOD_LEVEL, .lod_dword = lod, .tuple = G5P_TEX_TUPLE_COUPLED_TEMP_LEVEL,
      };
      uint8_t raw[G5P_USC_MAX];
      const unsigned n = g5p_usc_sample_register_lod(raw, &x);
      if (!n) {
         a9_tex_refuse(ctx, "the register-LOD sample was refused by the encoder");
         return false;
      }
      a9_tex_emit_raw(ctx, raw, n);
   } else {
   struct g5p_tex_ext x = {
      .dst_half = 2 * dst, .image_qword = image / 2, .sampler_qword = sampler / 2,
      .coord_dword = block, .dim = cube ? G5P_TEX_CUBE : array ? G5P_TEX_2D_ARRAY : G5P_TEX_2D, .mask = 0xf, .read = read,
      .lod = grad ? G5P_TEX_LOD_GRAD : G5P_TEX_LOD_LEVEL, .lod_dword = lod,
      .lod_shared = layered,
      .tuple = layered ? G5P_TEX_TUPLE_COUPLED_SHARED_LEVEL :
               grad ? G5P_TEX_TUPLE_COUPLED_GRAD : G5P_TEX_TUPLE_COUPLED_TEMP_LEVEL,
   };
   uint8_t raw[G5P_USC_MAX];
   const unsigned n = g5p_usc_sample_ext(raw, &x);
   if (!n) {
      a9_tex_refuse(ctx, "the compute sample recipe was refused by the encoder");
      return false;
   }
   a9_tex_emit_raw(ctx, raw, n);
   }
   g9p_wait(ctx->c);
   for (unsigned c = 0; c < 4; c++) {
      ctx->reg[key(&tex->def, c)] = dst + c;
      if (c >= tex->def.num_components || !ctx->last_use[key(&tex->def, c)])
         ctx->busy &= ~BITFIELD64_BIT(dst + c);
   }
   return true;
}
#define G9P_A9_TEX(ctx, tex) \
   do { if ((ctx)->out->a9 && (ctx)->stage == MESA_SHADER_COMPUTE) return a9_tex(ctx, tex); } while (0)
