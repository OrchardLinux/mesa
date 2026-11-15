/* SPDX-License-Identifier: MIT */
bool g9p_a9_compile_compute(nir_shader *nir, struct g9p_shader *out, void *mem_ctx)
{
   const nir_lower_compute_system_values_options csv = {
      .lower_cs_local_id_to_index = true, .global_id_is_32bit = true,
   };
   memset(out, 0, sizeof(*out));
   out->a9 = true;
   lower_front(nir);
   NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_mem_shared, glsl_get_natural_size_align_bytes);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_shared, nir_address_format_32bit_offset);
   if (nir->info.shared_size > 16384) {
      if (!out->a9_refusal[0])
         snprintf(out->a9_refusal, sizeof(out->a9_refusal), "shared allocation exceeds 16384 bytes");
      return false;
   }
   out->a9_local_dwords = 4 + DIV_ROUND_UP(nir->info.shared_size, 4);
   NIR_PASS(_, nir, nir_lower_compute_system_values, &csv);
   NIR_PASS(_, nir, nir_lower_undef_to_zero, NULL);
   if (!compile_back(nir, out, MESA_SHADER_COMPUTE, NULL, 0, false, false, mem_ctx))
      return false;
   /* the fixed common-store places of texture words (g9p_a9_tex.h) */
   bool sizes = false;
   for (unsigned i = 0; i < out->a9_num_uploads; i++)
      sizes |= out->a9_uploads[i].kind == G9P_A9_UP_SIZE;
   if (out->a9_num_uploads && out->num_buffers > 15) {
      if (!out->a9_refusal[0])
         snprintf(out->a9_refusal, sizeof(out->a9_refusal),
               "%u buffers with sampled textures (buffer words reach the texture words at common 32)",
               out->num_buffers);
      return false;
   }
   if (sizes && out->num_textures) {
      if (!out->a9_refusal[0])
         snprintf(out->a9_refusal, sizeof(out->a9_refusal),
               "a texture size query together with a storage image (both at common 40-47)");
      return false;
   }
   return true;
}
