/* SPDX-License-Identifier: MIT */
#ifdef PVR_APPLE_G5P
#include "../g5p/g5p_barrier.h"
#include "../g5p/g5p_atomic.h"
#include "../g5p/g5p_image.h"
/* Static threadgroup operands: A9 host-compiler forms (compute.md §4).
 * Reserve TG0/1 for IDs and TG2/3 for the future software-barrier counters.
 * CDM allocation covers system words plus the native shared block. */
/* S and RI address forms from compute-isa-shared-atomic.md §1.3.
 * RI reads a temporary half index for the last time (short parcel8800).
 * The uint32->half conversion uses the helper's explicitly inferred1189 form. */
static bool a9_shared(struct g9p_ctx *ctx, nir_intrinsic_instr *in)
{
   bool load = in->intrinsic == nir_intrinsic_load_shared;
   nir_src offset = in->src[load ? 0 : 1];
   unsigned bits = load ? in->def.bit_size : nir_src_bit_size(in->src[0]);
   unsigned comps = load ? in->def.num_components : nir_src_num_components(in->src[0]);
   if (!ctx->out->a9 || bits != 32 || comps != 1 || (nir_intrinsic_base(in) & 3))
      return false;
   unsigned word = 4 + nir_intrinsic_base(in) / 4;
   bool constant = nir_src_is_const(offset);
   int index = -1;
   if (constant) {
      unsigned byte = nir_src_as_uint(offset);
      if (byte & 3) return false;
      word += byte / 4;
   } else {
      int src = src_reg(ctx, nir_get_scalar(offset.ssa, 0));
      int shift = const_reg(ctx, 2);
      index = alloc_temp(ctx);
      if (src < 0 || shift < 0 || index < 0) return false;
      g9p_shift(ctx->c, index, src, shift, true);
      G9P_EMIT(ctx->c, 0x1189, 0x0300 | 2 * index,
               0x0101 | index << 10, 0x0120, 0x0fa1, 0x4000);
   }
   int reg = load ? alloc_reg(ctx) : src_reg(ctx, nir_get_scalar(in->src[0].ssa, 0));
   if (reg < 0 || word >= 65536) return false;
   unsigned w1 = 0x0100 | (!load ? 0x80 : 0) | 2 * reg;
   if (constant) {
      G9P_EMIT(ctx->c, 0x2314, w1, 0x8100 | (word & 255), 0x0200 | (word >> 8));
   } else {
      w1 |= (2 * index) << 9;
      if (word < 256) G9P_EMIT(ctx->c, 0x2314, w1, 0x8800 | word);
      else G9P_EMIT(ctx->c, 0x2314, w1, 0x8900 | (word & 255), word >> 8);
   }
   g9p_wait(ctx->c);
   release(ctx, &in->instr);
   if (load) ctx->reg[key(&in->def, 0)] = reg;
   return true;
}
/* Keep the election's local-index source live for every task barrier. */
static void a9_generate_init(struct g9p_ctx *ctx, nir_function_impl *impl)
{
   ctx->a9_barriers_remaining = 0;
   ctx->out->a9_device_fence = false;
   if (!ctx->out->a9) return;
   unsigned n = ctx->out->local_size[0] * ctx->out->local_size[1] * ctx->out->local_size[2];
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic) continue;
         nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
         if (in->intrinsic == nir_intrinsic_barrier &&
             (nir_intrinsic_memory_modes(in) & (nir_var_mem_ssbo | nir_var_mem_global | nir_var_image)))
            ctx->out->a9_device_fence = true;
         if (in->intrinsic == nir_intrinsic_ssbo_atomic_swap)
            ctx->busy |= ctx->pinned |= BITFIELD64_BIT(0);
         if (in->intrinsic == nir_intrinsic_barrier &&
             nir_intrinsic_execution_scope(in) == SCOPE_WORKGROUP)
            ctx->a9_barriers_remaining++;
      }
   }
   ctx->out->a9_has_barriers = n > 32 && ctx->a9_barriers_remaining;
   if (ctx->out->a9_has_barriers) ctx->busy |= ctx->pinned |= BITFIELD64_BIT(0);
}
static bool a9_barrier(struct g9p_ctx *ctx, nir_intrinsic_instr *in)
{
   if (!ctx->out->a9) return false;
   unsigned n = ctx->out->local_size[0] * ctx->out->local_size[1] * ctx->out->local_size[2];
   unsigned scope = nir_intrinsic_execution_scope(in);
   unsigned modes=nir_intrinsic_memory_modes(in);
   bool device=modes & (nir_var_mem_ssbo | nir_var_mem_global | nir_var_image);
   if ((modes & ~(nir_var_mem_shared | nir_var_mem_ssbo | nir_var_mem_global | nir_var_image)) ||
       (scope != SCOPE_NONE && scope != SCOPE_INVOCATION && scope != SCOPE_WORKGROUP)) {
      fprintf(stderr, "g5p: unsupported barrier memory mode or execution scope\n");
      return false;
   }
   int fence_pair=-1;
   if (device) {
      int zero=const_reg(ctx,0);
      fence_pair=alloc_pair(ctx);
      if (zero<0 || fence_pair<0 || fence_pair>30) return false;
      ctx->temps |= BITFIELD64_RANGE(fence_pair,2);
      /* Reserved common dwords56/57 hold this dispatch's writable scratch VA. */
      g9p_mov_uniform(ctx->c,fence_pair,56,zero);
      g9p_mov_uniform(ctx->c,fence_pair+1,57,zero);
   }
   if (scope == SCOPE_WORKGROUP) {
      unsigned N = ctx->a9_barriers_remaining--;
      if (!N || N > 31) return false;
      if (n > 32) {
         int r = alloc_temp(ctx), t = alloc_temp(ctx);
         if (r < 0 || t < 0) return false;
         /* Local size is fixed per pipeline: specialize Apple's target uniform. */
         g9p_mov_imm(ctx->c, t, DIV_ROUND_UP(n, 32) - 1);
         g5p_shared_barrier_fenced(ctx->c, r, 2 * t, 2, ctx->st, N, fence_pair);
         release(ctx, &in->instr);
         return true;
      }
      /* Apple's <=32 compiler form; cross-lane hardware qualification pending. */
   }
   if (device) {
      G9P_EMIT(ctx->c,0xc330,0x0100 | (unsigned)fence_pair<<10,0x0900,0x1000);
      release(ctx,&in->instr);
   }
   g9p_wait(ctx->c);
   return true;
}
static bool a9_ssbo_atomic(struct g9p_ctx *ctx, nir_intrinsic_instr *in)
{
   if (!ctx->out->a9 || ctx->stage != MESA_SHADER_COMPUTE ||
       in->def.bit_size != 32 || in->def.num_components != 1 ||
       nir_intrinsic_offset_shift(in) != 0) return false;
   unsigned group, op;
   switch (nir_intrinsic_atomic_op(in)) {
   case nir_atomic_op_iadd: group=0; op=0; break;
   case nir_atomic_op_isub: group=0; op=1; break;
   case nir_atomic_op_xchg: group=0; op=2; break;
   case nir_atomic_op_umin: group=1; op=0; break;
   case nir_atomic_op_imin: group=1; op=1; break;
   case nir_atomic_op_umax: group=1; op=2; break;
   case nir_atomic_op_imax: group=1; op=3; break;
   case nir_atomic_op_iand: group=2; op=0; break;
   case nir_atomic_op_ior: group=2; op=1; break;
   case nir_atomic_op_ixor: group=2; op=2; break;
   default: return false;
   }
   int slot=buffer_slot(ctx,in->src[0]);
   int off=src_reg(ctx,nir_get_scalar(in->src[1].ssa,0));
   int val=src_reg(ctx,nir_get_scalar(in->src[2].ssa,0));
   int zero=const_reg(ctx,0), pair=alloc_pair(ctx), dst=alloc_reg(ctx);
   if (slot<0 || off<0 || val<0 || zero<0 || pair<0 || dst<0 || pair>30 || dst>31) return false;
   ctx->temps |= BITFIELD64_RANGE(pair,2);
   g9p_addr(ctx->c,pair,2*slot,off,0,false);
   g5p_device_atomic(ctx->c,group,op,pair,dst,zero,val);
   release(ctx,&in->instr);
   ctx->reg[key(&in->def,0)]=dst;
   return true;
}
static bool a9_ssbo_cas(struct g9p_ctx *ctx, nir_intrinsic_instr *in)
{
   if (!ctx->out->a9 || ctx->stage != MESA_SHADER_COMPUTE ||
       in->def.bit_size != 32 || in->def.num_components != 1 ||
       nir_intrinsic_atomic_op(in) != nir_atomic_op_cmpxchg ||
       nir_intrinsic_offset_shift(in) != 0) return false;
   int slot=buffer_slot(ctx,in->src[0]);
   int off=src_reg(ctx,nir_get_scalar(in->src[1].ssa,0));
   int cmp=src_reg(ctx,nir_get_scalar(in->src[2].ssa,0));
   int val=src_reg(ctx,nir_get_scalar(in->src[3].ssa,0));
   int pair=alloc_pair(ctx), i=alloc_temp(ctx), dst=alloc_reg(ctx);
   if (slot<0 || off<0 || cmp<0 || val<0 || pair<0 || i<0 || dst<0 || pair>30 || dst>31 || i>31)
      return false;
   ctx->temps |= BITFIELD64_RANGE(pair,2);
   g9p_addr(ctx->c,pair,2*slot,off,0,false);
   g5p_device_cas(ctx->c,pair,dst,cmp,val,i,ctx->st);
   release(ctx,&in->instr);
   ctx->reg[key(&in->def,0)]=dst;
   return true;
}
static bool a9_image(struct g9p_ctx *ctx, nir_intrinsic_instr *in)
{
   bool store=in->intrinsic==nir_intrinsic_image_deref_store;
   if(!ctx->out->a9 || ctx->stage!=MESA_SHADER_COMPUTE ||
      nir_intrinsic_image_dim(in)!=GLSL_SAMPLER_DIM_2D || nir_intrinsic_image_array(in))return false;
   nir_deref_instr *deref=nir_src_as_deref(in->src[0]);
   nir_variable *v=nir_deref_instr_get_variable(deref);
   if(!v || deref->deref_type!=nir_deref_type_var || v->data.image.format!=PIPE_FORMAT_R32_UINT ||
      !nir_src_is_const(in->src[store?4:3]) || nir_src_as_uint(in->src[store?4:3])!=0)return false;
   if(ctx->out->num_textures && (ctx->out->textures[0].set!=v->data.descriptor_set ||
      ctx->out->textures[0].binding!=v->data.binding))return false;
   ctx->out->num_textures=1;
   ctx->out->textures[0].set=v->data.descriptor_set;
   ctx->out->textures[0].binding=v->data.binding;
   int co=alloc_pair(ctx);
   if(co<0 || co>30)return false;
   ctx->temps |= BITFIELD64_RANGE(co,2);
   for(unsigned j=0;j<2;j++) {
      int r=src_reg(ctx,nir_get_scalar(in->src[1].ssa,j));
      if(r<0)return false;
      g9p_alu2(ctx->c,G9P_OR,co+j,r,r,false);
   }
   int data=alloc_range(ctx,4);
   if(data<0 || data>28)return false;
   if(store) {
      if(nir_src_bit_size(in->src[3])!=32 || nir_src_num_components(in->src[3])!=4)return false;
      ctx->temps |= BITFIELD64_RANGE(data,4);
      for(unsigned j=0;j<4;j++) {
         int r=src_reg(ctx,nir_get_scalar(in->src[3].ssa,j));
         if(r<0)return false;
         g9p_alu2(ctx->c,G9P_OR,data+j,r,r,false);
      }
   } else if(in->def.bit_size!=32 || in->def.num_components>4)return false;
   g5p_image_raw32(ctx->c,store,data,co,40,31);
   release(ctx,&in->instr);
   if(!store) for(unsigned j=0;j<4;j++) {
      ctx->reg[key(&in->def,j)]=data+j;
      if(j>=in->def.num_components || !ctx->last_use[key(&in->def,j)])ctx->busy &= ~BITFIELD64_BIT(data+j);
   }
   return true;
}
#define G9P_A9_INTRINSICS \
   if (ctx->out->a9) { switch (intr->intrinsic) { \
   case nir_intrinsic_image_deref_store: case nir_intrinsic_image_deref_load: return a9_image(ctx, intr); \
   case nir_intrinsic_ssbo_atomic_swap: return a9_ssbo_cas(ctx, intr); \
   case nir_intrinsic_ssbo_atomic: return a9_ssbo_atomic(ctx, intr); \
   case nir_intrinsic_load_shared: case nir_intrinsic_store_shared: return a9_shared(ctx, intr); \
   case nir_intrinsic_barrier: return a9_barrier(ctx, intr); \
   default: break; } }
#else
#define G9P_A9_INTRINSICS
#endif
