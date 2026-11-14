/* SPDX-License-Identifier: MIT
 * Generated-by: LLM
 * Keep BUF semantics in NIR. backend1 carries the native 2D folded coordinate;
 * txs reads the logical descriptor range. Invalid indices select valid texel0
 * before the hardware access and return all-zero afterwards.
 */
#ifndef G5P_TEXEL_NIR_H
#define G5P_TEXEL_NIR_H
#include "g5p_texel_defs.h"
static bool pvr_g5p_tbo_lower_instr(nir_builder *b, nir_instr *instr, void *data)
{
   (void)data;
   if (instr->type != nir_instr_type_tex) return false;
   nir_tex_instr *t = nir_instr_as_tex(instr);
   if (t->sampler_dim != GLSL_SAMPLER_DIM_BUF || t->op != nir_texop_txf ||
       (t->backend_flags & PVR_G5P_TBO_NIR_FOLDED) || t->is_array || t->is_shadow) return false;
   nir_deref_instr *d = nir_get_tex_deref(t, nir_tex_src_texture_deref);
   if (!d || d->deref_type != nir_deref_type_var) return false;
   b->cursor = nir_before_instr(instr);
   nir_def *size = nir_txs(b, .texture_deref = d, .dest_type = nir_type_int32);
   nir_def *index = nir_get_tex_src(t, nir_tex_src_coord);
   nir_def *valid = nir_ult(b, index, size);
   nir_def *safe = nir_bcsel(b, valid, index, nir_imm_int(b, 0));
   nir_def *xy = nir_vec2(b, nir_iand_imm(b, safe, PVR_G5P_TBO_WIDTH - 1), nir_ushr_imm(b, safe, PVR_G5P_TBO_SHIFT));
   nir_src_rewrite(&t->src[nir_tex_instr_src_index(t, nir_tex_src_coord)].src, nir_imm_int(b, 0));
   nir_tex_instr_add_src(t, nir_tex_src_backend1, xy);
   t->backend_flags |= PVR_G5P_TBO_NIR_FOLDED;
   b->cursor = nir_after_instr(instr);
   nir_def *masked = nir_bcsel(b, valid, &t->def, nir_imm_zero(b, t->def.num_components, t->def.bit_size));
   nir_def_rewrite_uses_after(&t->def, masked);
   return true;
}
static inline void pvr_g5p_tbo_lower(nir_shader *nir)
{
   if (nir && pvr_g5p_tbo_probe())
      nir_shader_instructions_pass(nir, pvr_g5p_tbo_lower_instr, nir_metadata_control_flow, NULL);
}
#endif
