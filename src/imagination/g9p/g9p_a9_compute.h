/* SPDX-License-Identifier: MIT */
#ifdef PVR_APPLE_G5P
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
      fprintf(stderr, "g5p: shared allocation exceeds16384 bytes\n");
      return false;
   }
   out->a9_local_dwords = 4 + DIV_ROUND_UP(nir->info.shared_size, 4);
   NIR_PASS(_, nir, nir_lower_compute_system_values, &csv);
   NIR_PASS(_, nir, nir_lower_undef_to_zero, NULL);
   return compile_back(nir, out, MESA_SHADER_COMPUTE, NULL, 0, false, false, mem_ctx);
}
#endif
