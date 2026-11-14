/* SPDX-License-Identifier: MIT
 * Generated-by: LLM
 * Evaluate the actual lowered NIR after substituting input constants;
 * compare with independent integer references, not a duplicate SWAR loop.
 */
#include "g5p_nir_integer.h"
#include "compiler/glsl_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"CHECK failed %s:%d: %s\n",__FILE__,__LINE__,#x); abort(); } } while(0)
static const nir_shader_compiler_options options = {
   .lower_bitfield_extract=true, .lower_bitfield_insert=true,
   .lower_bit_count=true, .lower_bitfield_reverse=true,
   .lower_ifind_msb=true, .lower_find_lsb=true,
};
static void optimize(nir_shader *s)
{
   bool progress;
   do {
      progress=false;
      NIR_PASS(progress,s,nir_opt_constant_folding);
      NIR_PASS(progress,s,nir_opt_copy_prop);
      NIR_PASS(progress,s,nir_opt_dce);
      NIR_PASS(progress,s,nir_opt_algebraic);
   } while(progress);
}
static uint32_t reference(unsigned op, uint32_t x, uint32_t y, unsigned off, unsigned bits)
{
   uint32_t mask=bits==32 ? UINT32_MAX : (UINT32_C(1)<<bits)-1;
   uint32_t field=bits ? (x >> (off&31)) & mask : 0;
   switch(op) {
   case 0: return field;
   case 1: return bits && bits<32 && (field & (UINT32_C(1)<<(bits-1))) ? field|~mask : field;
   case 2: return bits==32 ? y : (x & ~(mask << (off&31))) | ((y & mask) << (off&31));
   case 3: return __builtin_popcount(x);
   case 4: {uint32_t r=0;for(unsigned i=0;i<32;i++)r|=((x>>i)&1u)<<(31-i);return r;}
   case 5: return x ? 31-__builtin_clz(x) : UINT32_MAX;
   case 6: {uint32_t t=(x&0x80000000u)?~x:x;return t?31-__builtin_clz(t):UINT32_MAX;}
   default:return x?__builtin_ctz(x):UINT32_MAX;
   }
}
int main(void)
{
   glsl_type_singleton_init_or_ref();
   nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_VERTEX,&options,"g5p-integer-semantics");
   nir_def *v[4];
   for(unsigned i=0;i<4;i++)v[i]=nir_load_uniform(&b,1,32,nir_imm_int(&b,0),.base=i,.range=1);
   nir_def *outputs[]={nir_ubitfield_extract(&b,v[0],v[2],v[3]),nir_ibitfield_extract(&b,v[0],v[2],v[3]),nir_bitfield_insert(&b,v[0],v[1],v[2],v[3]),nir_bit_count(&b,v[0]),nir_bitfield_reverse(&b,v[0]),nir_ufind_msb(&b,v[0]),nir_ifind_msb(&b,v[0]),nir_find_lsb(&b,v[0])};
   for(unsigned i=0;i<8;i++)nir_store_output(&b,outputs[i],nir_imm_int(&b,0),.base=i,.write_mask=1);
   optimize(b.shader);
   NIR_PASS(_,b.shader,nir_shader_alu_pass,g5p_lower_msb_instr,nir_metadata_control_flow,NULL);
   NIR_PASS(_,b.shader,nir_lower_alu);
   optimize(b.shader);
   nir_validate_shader(b.shader,"lowered integer template");
   const uint32_t edges[]={0,1,2,0x7fffffffu,0x80000000u,0xffffffffu,0xaaaaaaaau,0x55555555u};
   uint32_t rng=0x96af38d1u;unsigned tested=0;
   for(unsigned n=0;n<4096;n++) {
      rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;
      uint32_t x=n<8?edges[n]:rng,y=~rng;
      unsigned off=n%33,bits=(n/33)%(33-off);
      if(n<8){off=0;bits=n%2?32:0;}
      uint32_t inputs[]={x,y,off,bits};
      nir_shader *s=nir_shader_clone(NULL,b.shader);
      nir_foreach_function_impl(impl,s) {
         nir_foreach_block(block,impl) {
            nir_foreach_instr_safe(ins,block) {
               if(ins->type!=nir_instr_type_intrinsic)continue;
               nir_intrinsic_instr *in=nir_instr_as_intrinsic(ins);
               if(in->intrinsic!=nir_intrinsic_load_uniform)continue;
               nir_builder cb=nir_builder_create(impl);cb.cursor=nir_before_instr(ins);
               nir_def_replace(&in->def,nir_imm_int(&cb,inputs[nir_intrinsic_base(in)]));
            }
         }
      }
      optimize(s);unsigned seen=0;
      nir_foreach_function_impl(impl,s) {
         nir_foreach_block(block,impl) {
            nir_foreach_instr(ins,block) {
               if(ins->type!=nir_instr_type_intrinsic)continue;
               nir_intrinsic_instr *in=nir_instr_as_intrinsic(ins);
               if(in->intrinsic!=nir_intrinsic_store_output)continue;
               unsigned op=nir_intrinsic_base(in);CHECK(nir_src_is_const(in->src[0]));
               uint32_t got=nir_src_as_uint(in->src[0]),want=reference(op,x,y,off,bits);
               if(got!=want){fprintf(stderr,"op%u x=%08x y=%08x offset=%u bits=%u got=%08x want=%08x\n",op,x,y,off,bits,got,want);abort();}
               seen++;
            }
         }
      }
      CHECK(seen==8);tested+=seen;ralloc_free(s);
   }
   ralloc_free(b.shader);glsl_type_singleton_decref();
   printf("INTEGER LOWERING PASS %u exact lowered-NIR results, zero/full width and signed edges\n",tested);
   return 0;
}
