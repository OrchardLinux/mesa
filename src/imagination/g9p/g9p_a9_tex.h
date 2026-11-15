/* SPDX-License-Identifier: MIT */
#include "../g5p/g5p_usc.h"
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
   if (!ctx->out->a9_refusal[0])
      if (!ctx->out->a9_refusal[0])
      snprintf(ctx->out->a9_refusal, sizeof(ctx->out->a9_refusal), "%s", why);
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
   if (tex->sampler_dim != GLSL_SAMPLER_DIM_2D || tex->is_array || tex->is_shadow) {
      a9_tex_refuse(ctx, tex->is_shadow ? "a depth-compare sample in a compute shader" :
                         "only 2D (not array, cube or 3D) samples in compute shaders so far");
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
   const int block = alloc_range(ctx, 2);
   if (block < 0 || block > 30)
      return false;
   ctx->temps |= BITFIELD64_RANGE(block, 2);
   for (unsigned c = 0; c < 2; c++)
      if (!a9_tex_copy(ctx, block + c, tex->src[ci].src.ssa, c))
         return false;

   /* level (f16 for samples, u32 for reads) or gradients */
   int lod;
   if (grad) {
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
   release(ctx, &tex->instr);
   const int dst = alloc_range(ctx, 4);
   if (dst < 0 || dst > 28)
      return false;

   struct g5p_tex_ext x = {
      .dst_half = 2 * dst, .image_qword = image / 2, .sampler_qword = sampler / 2,
      .coord_dword = block, .dim = G5P_TEX_2D, .mask = 0xf, .read = read,
      .lod = grad ? G5P_TEX_LOD_GRAD : G5P_TEX_LOD_LEVEL, .lod_dword = lod,
      .tuple = grad ? G5P_TEX_TUPLE_COUPLED_GRAD : G5P_TEX_TUPLE_COUPLED_TEMP_LEVEL,
   };
   uint8_t raw[G5P_USC_MAX];
   const unsigned n = g5p_usc_sample_ext(raw, &x);
   if (!n) {
      a9_tex_refuse(ctx, "the compute sample recipe was refused by the encoder");
      return false;
   }
   a9_tex_emit_raw(ctx, raw, n);
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
