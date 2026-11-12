/* SPDX-License-Identifier: MIT */
#ifdef PVR_APPLE_G5P
/* Direct dispatches with one Z workgroup have WorkgroupId.z == 0.
 * Specialize only when the paired dispatch guard is enabled by this flag.
 * LocalInvocationId.z remains computed normally, including local_size_z>1. */
static bool a9_unit_z(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
   if (in->intrinsic != nir_intrinsic_load_workgroup_id ||
       !(nir_def_components_read(&in->def) & 4))
      return false;
   struct g9p_shader *out = data;
   b->cursor = nir_after_instr(&in->instr);
   nir_def *v = nir_vec3(b, nir_channel(b, &in->def, 0),
                        nir_channel(b, &in->def, 1), nir_imm_int(b, 0));
   nir_def_rewrite_uses_after(&in->def, v);
   out->a9_unit_z = true;
   return true;
}

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
   const char *unit_z = getenv("PVR_G5P_COMPUTE_UNIT_Z");
   if (unit_z && !strcmp(unit_z, "1"))
      nir_shader_intrinsics_pass(nir, a9_unit_z, nir_metadata_control_flow, out);
   NIR_PASS(_, nir, nir_lower_undef_to_zero, NULL);
   return compile_back(nir, out, MESA_SHADER_COMPUTE, NULL, 0, false, false, mem_ctx);
}
#endif
