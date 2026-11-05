/* SPDX-License-Identifier: MIT */
#ifdef PVR_APPLE_G5P
/* Lower bounded dynamic shared addresses into native constant-offset TG
 * operations under normal lane masks. This avoids guessing the unresolved
 * indirect-TG operand fields and does not substitute global-memory storage. */
static bool a9_lower_dynamic_shared(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
   bool load = in->intrinsic == nir_intrinsic_load_shared;
   if (!load && in->intrinsic != nir_intrinsic_store_shared) return false;
   unsigned index = load ? 0 : 1;
   if (nir_src_is_const(in->src[index])) return false;
   unsigned comps = load ? in->def.num_components : nir_src_num_components(in->src[0]);
   unsigned bits = load ? in->def.bit_size : nir_src_bit_size(in->src[0]);
   if (comps != 1 || bits != 32) return false;
   b->cursor = nir_before_instr(&in->instr);
   unsigned size = *(unsigned *)data;
   nir_def *off = nir_iadd_imm(b, in->src[index].ssa, nir_intrinsic_base(in));
   nir_def *value = nir_imm_int(b, 0);
   for (unsigned byte = 0; byte < size; byte += 4) {
      nir_push_if(b, nir_ieq_imm(b, off, byte));
      if (load) {
         nir_def *v = nir_load_shared(b, 1, 32, nir_imm_int(b, byte), .align_mul = 4, .access = nir_intrinsic_access(in));
         nir_push_else(b, NULL);
         nir_pop_if(b, NULL);
         value = nir_if_phi(b, v, value);
      } else {
         nir_store_shared(b, in->src[0].ssa, nir_imm_int(b, byte),
                          .write_mask = 1, .align_mul = 4, .access = nir_intrinsic_access(in));
         nir_pop_if(b, NULL);
      }
   }
   if (load) nir_def_replace(&in->def, value);
   else nir_instr_remove(&in->instr);
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
   if (nir->info.shared_size > 32) {
      fprintf(stderr, "g5p: shared allocation >32 bytes requires CDM allocation support\n");
      return false;
   }
   unsigned shared_size = nir->info.shared_size;
   if (shared_size)
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, a9_lower_dynamic_shared, nir_metadata_none, &shared_size);
   NIR_PASS(_, nir, nir_lower_compute_system_values, &csv);
   return compile_back(nir, out, MESA_SHADER_COMPUTE, NULL, 0, mem_ctx);
}
#endif
