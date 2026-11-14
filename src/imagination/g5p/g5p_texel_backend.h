/* SPDX-License-Identifier: MIT
 * Generated-by: LLM
 * Admission of the folded buffer form only. Ordinary vertex textures retain
 * their independently qualified route. No storage-texel admission here.
 */
#ifndef G5P_TEXEL_BACKEND_H
#define G5P_TEXEL_BACKEND_H
#include "g5p_texel_defs.h"
static const char *g5p_tbo_shape_refusal(const nir_tex_instr *t)
{
   if (!pvr_g5p_tbo_probe()) return "texel-buffer path requires qualification (PVR_G5P_TEXEL_BUFFER=probe)";
   nir_deref_instr *d = nir_get_tex_deref(t, nir_tex_src_texture_deref);
   if (!d || d->deref_type != nir_deref_type_var)
      return "texel-buffer descriptor arrays or bindless handles not qualified";
   if (t->def.bit_size != 32) return "texel-buffer result is not 32-bit";
   if (t->is_array || t->is_shadow) return "array or shadow texel buffer";
   if (t->op == nir_texop_txs) return t->def.num_components == 1 ? NULL : "texel-buffer size width";
   if (t->op != nir_texop_txf || t->def.num_components != 4) return "texel-buffer operation other than four-channel fetch";
   if (!(t->backend_flags & PVR_G5P_TBO_NIR_FOLDED)) return "texel-buffer fetch lacks robust folded coordinate";
   for (unsigned i = 0; i < t->num_srcs; i++) {
      switch (t->src[i].src_type) {
      case nir_tex_src_coord: case nir_tex_src_texture_deref: case nir_tex_src_backend1: break;
      case nir_tex_src_lod:
         if (!nir_src_is_const(t->src[i].src) || nir_src_as_uint(t->src[i].src) != 0)
            return "nonzero texel-buffer LOD";
         break;
      default: return "texel-buffer source outside the samplerless fold";
      }
   }
   return NULL;
}
#endif
