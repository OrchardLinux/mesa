/* SPDX-License-Identifier: MIT */
/* One invocation is the whole subgroup on G5P. */
static bool
a9_lower_subgroups_size1(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
   b->cursor = nir_before_instr(&in->instr);
   nir_def *v, *bit, *parts[NIR_MAX_VEC_COMPONENTS];
   unsigned n = in->def.num_components, bits = in->def.bit_size;
   switch (in->intrinsic) {
   case nir_intrinsic_load_subgroup_size:
      v = nir_imm_int(b, 1); break;
   case nir_intrinsic_load_subgroup_invocation:
   case nir_intrinsic_first_invocation:
   case nir_intrinsic_last_invocation:
      v = nir_imm_int(b, 0); break;
   case nir_intrinsic_load_num_subgroups:
      v = nir_imm_int(b, b->shader->info.workgroup_size[0] *
                        b->shader->info.workgroup_size[1] *
                        b->shader->info.workgroup_size[2]); break;
   case nir_intrinsic_load_subgroup_id:
      v = nir_load_local_invocation_index(b); break;
   case nir_intrinsic_load_subgroup_eq_mask:
   case nir_intrinsic_load_subgroup_ge_mask:
   case nir_intrinsic_load_subgroup_le_mask:
   case nir_intrinsic_ballot:
      parts[0] = in->intrinsic == nir_intrinsic_ballot ?
                    nir_b2iN(b, in->src[0].ssa, bits) : nir_imm_intN_t(b, 1, bits);
      for (unsigned j = 1; j < n; j++) parts[j] = nir_imm_intN_t(b, 0, bits);
      v = nir_vec(b, parts, n); break;
   case nir_intrinsic_load_subgroup_gt_mask:
   case nir_intrinsic_load_subgroup_lt_mask:
   case nir_intrinsic_ballot_bit_count_exclusive:
      v = nir_imm_zero(b, n, bits); break;
   case nir_intrinsic_vote_ieq:
   case nir_intrinsic_vote_feq:
   case nir_intrinsic_elect:
      v = nir_imm_true(b); break;
   case nir_intrinsic_vote_any:
   case nir_intrinsic_vote_all:
   case nir_intrinsic_read_invocation:
   case nir_intrinsic_read_first_invocation:
   case nir_intrinsic_shuffle:
   case nir_intrinsic_shuffle_xor:
   case nir_intrinsic_shuffle_up:
   case nir_intrinsic_shuffle_down:
   case nir_intrinsic_rotate:
   case nir_intrinsic_quad_broadcast:
   case nir_intrinsic_quad_swap_horizontal:
   case nir_intrinsic_quad_swap_vertical:
   case nir_intrinsic_quad_swap_diagonal:
   case nir_intrinsic_quad_vote_any:
   case nir_intrinsic_quad_vote_all:
   case nir_intrinsic_reduce:
   case nir_intrinsic_inclusive_scan:
      v = in->src[0].ssa; break;
   case nir_intrinsic_inverse_ballot:
   case nir_intrinsic_ballot_bitfield_extract:
   case nir_intrinsic_ballot_bit_count_reduce:
   case nir_intrinsic_ballot_bit_count_inclusive:
   case nir_intrinsic_ballot_find_lsb:
   case nir_intrinsic_ballot_find_msb:
      bit = nir_iand_imm(b, nir_channel(b, in->src[0].ssa, 0), 1);
      if (in->intrinsic == nir_intrinsic_inverse_ballot)
         v = nir_ine_imm(b, bit, 0);
      else if (in->intrinsic == nir_intrinsic_ballot_bitfield_extract)
         v = nir_iand(b, nir_ieq_imm(b, in->src[1].ssa, 0), nir_ine_imm(b, bit, 0));
      else if (in->intrinsic == nir_intrinsic_ballot_find_lsb ||
               in->intrinsic == nir_intrinsic_ballot_find_msb)
         v = nir_bcsel(b, nir_ine_imm(b, bit, 0), nir_imm_int(b, 0), nir_imm_int(b, -1));
      else
         v = nir_u2uN(b, bit, bits);
      break;
   case nir_intrinsic_exclusive_scan: {
      nir_const_value identity = nir_alu_binop_identity(nir_intrinsic_reduction_op(in), bits);
      nir_def *scalar = nir_build_imm(b, 1, bits, &identity);
      for (unsigned j = 0; j < n; j++) parts[j] = scalar;
      v = nir_vec(b, parts, n); break;
   }
   default: return false;
   }
   nir_def_replace(&in->def, v);
   return true;
}

/* Apple's shared-memory size codes require size-aligned 1/2/4-dword runs. */
static nir_mem_access_size_align
a9_shared_size_align(nir_intrinsic_op op, uint8_t bytes, uint8_t bit_size,
                     uint32_t align_mul, uint32_t align_offset,
                     bool offset_is_const, enum gl_access_qualifier access,
                     const void *data)
{
   unsigned align = nir_combined_align(align_mul, align_offset);
   if (bit_size != 32 || align < 4)
      return (nir_mem_access_size_align) { 1, bit_size, bit_size / 8 };
   unsigned comps = bytes / 4;
   unsigned n = comps >= 4 && align >= 16 ? 4 : comps >= 2 && align >= 8 ? 2 : 1;
   return (nir_mem_access_size_align) { n, 32, 4 * n, nir_mem_access_shift_method_scalar };
}

/* Specialise storage writes before ALU lowering. Two data components mark
 * Apple's packed 0x18 form; raw 0x1a always has four dwords. Cube image
 * coordinates are already x,y,face (or 6*cube+face) in SPIR-V/NIR. */
static bool
a9_pack_image_store(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
   if (in->intrinsic != nir_intrinsic_image_deref_store)
      return false;
   nir_variable *v = nir_deref_instr_get_variable(nir_src_as_deref(in->src[0]));
   if (!v || nir_src_bit_size(in->src[3]) != 32)
      return false;
   b->cursor = nir_before_instr(&in->instr);
   nir_def *value = in->src[3].ssa, *lo, *hi = nir_imm_int(b, 0);
   switch (v->data.image.format) {
   case PIPE_FORMAT_R32_UINT: case PIPE_FORMAT_R32_SINT: case PIPE_FORMAT_R32_FLOAT:
      value = nir_vec4(b, nir_channel(b, value, 0), hi, hi, hi);
      break;
   case PIPE_FORMAT_R32G32B32A32_UINT: case PIPE_FORMAT_R32G32B32A32_SINT:
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
      return false;
   case PIPE_FORMAT_R8G8B8A8_UNORM:
      lo = nir_pack_unorm_4x8(b, value);
      value = nir_vec2(b, lo, hi);
      break;
   case PIPE_FORMAT_R16G16B16A16_FLOAT:
      hi = nir_pack_half_2x16(b, nir_channels(b, value, 0xc));
      FALLTHROUGH;
   case PIPE_FORMAT_R16G16_FLOAT:
      lo = nir_pack_half_2x16(b, nir_channels(b, value, 0x3));
      value = nir_vec2(b, lo, hi);
      break;
   case PIPE_FORMAT_R16_FLOAT:
      lo = nir_pack_half_2x16(b, nir_vec2(b, nir_channel(b, value, 0), nir_imm_float(b, 0)));
      value = nir_vec2(b, lo, hi);
      break;
   default:
      return false; /* a9_image() records the format by name */
   }
   nir_src_rewrite(&in->src[3], value);
   in->num_components = value->num_components;
   return true;
}

/* The texture unit takes an integer array layer even for filtered samples. */
static bool
a9_lower_array_layer(nir_builder *b, nir_tex_instr *tex, void *data)
{
   if (tex->op != nir_texop_txl || tex->sampler_dim != GLSL_SAMPLER_DIM_2D || !tex->is_array)
      return false;
   int ci = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (ci < 0) return false;
   b->cursor = nir_before_instr(&tex->instr);
   nir_def *co = tex->src[ci].src.ssa;
   nir_def *layer = nir_f2u32(b, nir_fround_even(b, nir_fmax(b, nir_channel(b, co, 2),
                                                             nir_imm_float(b, 0))));
   nir_src_rewrite(&tex->src[ci].src,
                  nir_vec3(b, nir_channel(b, co, 0), nir_channel(b, co, 1), layer));
   return true;
}

bool g9p_a9_compile_compute(nir_shader *nir, struct g9p_shader *out, void *mem_ctx)
{
   nir->options = &nir_options_a9;
   const nir_lower_compute_system_values_options csv = {
      .lower_cs_local_id_to_index = true, .global_id_is_32bit = true,
   };
   memset(out, 0, sizeof(*out));
   out->a9 = true;
   lower_front(nir);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, a9_lower_subgroups_size1,
            nir_metadata_control_flow, NULL);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, a9_pack_image_store,
            nir_metadata_control_flow, NULL);
   NIR_PASS(_, nir, nir_lower_alu);
   NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_mem_shared, glsl_get_natural_size_align_bytes);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_shared, nir_address_format_32bit_offset);
   const nir_lower_mem_access_bit_sizes_options shared_opts = {
      .modes = nir_var_mem_shared, .callback = a9_shared_size_align,
   };
   NIR_PASS(_, nir, nir_lower_mem_access_bit_sizes, &shared_opts);
   if (nir->info.shared_size > 16384) {
      if (!out->a9_refusal[0])
         snprintf(out->a9_refusal, sizeof(out->a9_refusal), "shared allocation exceeds 16384 bytes");
      return false;
   }
   out->a9_local_dwords = 4 + DIV_ROUND_UP(nir->info.shared_size, 4);
   NIR_PASS(_, nir, nir_shader_tex_pass, a9_lower_array_layer, nir_metadata_control_flow, NULL);
   NIR_PASS(_, nir, nir_lower_compute_system_values, &csv);
   NIR_PASS(_, nir, nir_lower_undef_to_zero, NULL);
   if (!compile_back(nir, out, MESA_SHADER_COMPUTE, NULL, 0, false, false, mem_ctx))
      return false;
   /* the fixed common-store places of texture words (g9p_a9_tex.h) */
   bool sizes = false, lods = false;
   for (unsigned i = 0; i < out->a9_num_uploads; i++) {
      sizes |= out->a9_uploads[i].kind == G9P_A9_UP_SIZE;
      lods |= out->a9_uploads[i].kind == G9P_A9_UP_LOD;
   }
   if (lods && out->num_buffers > 13) {
      if (!out->a9_refusal[0])
         snprintf(out->a9_refusal, sizeof(out->a9_refusal),
               "more than 13 buffers with shared texture levels (common 28-30)");
      return false;
   }
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
