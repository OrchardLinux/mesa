/* SPDX-License-Identifier: MIT
 * G5P backend consuming Mesa PCO IR before Rogue scheduling/grouping.
 * Implements physical MOVI32 + UVSW output and task finalization only.
 * Source of verified bit layouts: g5p_isa.py and isa-g5p.md.
 */
#include "pco.h"
#include "pco_g5p.h"
#include "g5p_usc.h"
#include "util/half_float.h"
#include "pco_builder.h"
#include "pco_internal.h"
#include "util/ralloc.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct binary { uint8_t data[1024]; unsigned size; };

#include <setjmp.h>
/* Per-thread escape only unwinds this bounded encoder; never exits the app. */
static _Thread_local jmp_buf reject;
static void require(bool ok, const char *why)
{ if(!ok) { fprintf(stderr,"G5P PCO rejects: %s\n",why); longjmp(reject,1); } }

static void put(struct binary *b, unsigned v)
{
   require(b->size < sizeof(b->data), "binary too large");
   b->data[b->size++] = v;
}

static void word(struct binary *b, unsigned v)
{
   put(b, v); put(b, v >> 8);
}

static void movimm(struct binary *b, unsigned dst_half, unsigned bank, uint32_t v)
{
   static const uint8_t bytes[32] = {4,4,4,4,4,4,6,6,6,1,6,7,7,7,6,6,8,8,8,8,8,8,8,8,7,9,9,9,9,9,9,9};
   static const uint8_t bits[32] = {1,2,3,4,5,6,1,2,3,5,4,4,5,6,6,7,0,1,2,3,4,5,6,7,7,1,2,3,4,5,6,7};
   uint8_t enc[10] = {0x98,1,dst_half,(bank << 1)|1,1,1,0x20,1,0,1};
   bool wide = v > 0xffff;
   if (wide) enc[6] |= 1;
   for (unsigned i = 0; i < 32; ++i)
      if (v & (1u << i)) enc[bytes[i]] |= 1u << bits[i];
   for (unsigned i = 0; i < (wide ? 10 : 8); ++i) put(b, enc[i]);
   word(b, 0x0154); word(b, 0x0038);
}

static void vout(struct binary *b, unsigned src, unsigned slot, bool last)
{
   require(src < 128 && slot < 256, "vertex operand range");
   if (!last && src == 0 && slot == 0) { word(b, 0x141e); return; }
   word(b, last ? 0x155e : 0x151e);
   word(b, 2 * src | (last ? 0xc100 : slot ? 0x0100 : 0));
   if (last || slot) word(b, slot);
}

static void encode(pco_shader *shader, struct binary *out)
{
   unsigned written = 0;
   bool ended = false;
   require(!shader->is_grouped, "stock Rogue instruction groups");
   pco_foreach_func_in_shader(func, shader) {
      pco_foreach_instr_in_func(ins, func) {
         require(!ended, "instruction after task end");
         require(pco_instr_has_default_exec(ins), "conditional execution");
         switch (ins->op) {
         case PCO_OP_MOVI32: {
            require(shader->stage == MESA_SHADER_FRAGMENT, "non-fragment movi32");
            require(pco_ref_is_imm(ins->src[0]) && pco_ref_is_hwreg(ins->dest[0]), "nonphysical move");
            require(pco_ref_get_reg_class(ins->dest[0]) == PCO_REG_CLASS_PIXOUT, "move bank");
            require(pco_instr_get_rpt(ins) == 1 && !pco_instr_get_atom(ins) && !pco_instr_get_olchk(ins), "move modifiers");
            unsigned d = pco_ref_get_reg_index(ins->dest[0]);
            require(d == written && d < 4, "output words must be contiguous o0..o3");
            movimm(out, d * 2, 0, pco_ref_get_imm(ins->src[0]));
            ++written;
            if (pco_instr_get_end(ins)) {
               require(written == 2 || written == 4, "fragment word count");
               word(out, 0x011c | (written << 12)); word(out, 0x0800); word(out, 0x8000);
               ended = true;
            }
            break;
         }
         case PCO_OP_UVSW_WRITE:
         case PCO_OP_UVSW_WRITE_EMIT_ENDTASK: {
            require(shader->stage == MESA_SHADER_VERTEX, "non-vertex UVSW");
            require(pco_ref_is_hwreg(ins->src[0]) && pco_ref_get_reg_class(ins->src[0]) == PCO_REG_CLASS_TEMP,
                    "vertex PDS input must be a physical temp");
            require(pco_ref_is_imm(ins->src[1]), "indirect output slot");
            if (ins->op == PCO_OP_UVSW_WRITE)
               require(pco_instr_get_rpt(ins) == 1, "repeated UVSW");
            unsigned slot = pco_ref_get_imm(ins->src[1]);
            bool last = ins->op == PCO_OP_UVSW_WRITE_EMIT_ENDTASK;
            if (last) {
               require(out->size + 6 <= 64, "vertex body exceeds characterized block");
               unsigned delta = 64 - out->size;
               word(out, 0x0100);
               word(out, (delta & 0xfe) | ((delta >> 8) & 1) | (((delta >> 9) << 1 | 1) << 8));
               word(out, 0);
               while (out->size < 64) word(out, 0xc000);
            }
            vout(out, pco_ref_get_reg_index(ins->src[0]), slot, last);
            ended = last;
            break;
         }
         default: require(false, "unsupported PCO opcode");
         }
      }
   }
   require(ended, "missing task end");
}


/* Shapes with an established physical ABI (round16 pass/solid, round17 vc
 * capture). Returns the shape id (pco.h), 0 for anything else; all other
 * shaders go to the round-4 SSA backend, never stock Rogue grouping. */
static bool whole_vec4_input(nir_def *def, unsigned location)
{
   nir_instr *parent = nir_def_instr(def);
   if (parent->type != nir_instr_type_intrinsic) return false;
   nir_intrinsic_instr *in = nir_instr_as_intrinsic(parent);
   return in->intrinsic == nir_intrinsic_load_input && in->num_components == 4 &&
          !nir_intrinsic_component(in) && nir_src_is_const(in->src[0]) && !nir_src_as_uint(in->src[0]) &&
          nir_intrinsic_io_semantics(in).location == location;
}

static bool smooth_varying(nir_def *def, unsigned location)
{
   nir_instr *parent = nir_def_instr(def);
   if (parent->type != nir_instr_type_intrinsic) return false;
   nir_intrinsic_instr *in = nir_instr_as_intrinsic(parent);
   if (in->intrinsic != nir_intrinsic_load_interpolated_input || in->num_components != 4 ||
       nir_intrinsic_component(in) || !nir_src_is_const(in->src[1]) || nir_src_as_uint(in->src[1]) ||
       nir_intrinsic_io_semantics(in).location != location)
      return false;
   nir_instr *bary = nir_def_instr(in->src[0].ssa);
   if (bary->type != nir_instr_type_intrinsic) return false;
   nir_intrinsic_instr *b = nir_instr_as_intrinsic(bary);
   return b->intrinsic == nir_intrinsic_load_barycentric_pixel &&
          (nir_intrinsic_interp_mode(b) == INTERP_MODE_NONE ||
           nir_intrinsic_interp_mode(b) == INTERP_MODE_SMOOTH);
}

/* The varying fragment program: iterate varying slot 1 as packed f16 into
 * o0..o1, wait for it, emit the 2 words, end (g5p_usc.h's encoders). */
static bool encode_varying_fs(struct binary *out)
{
   uint8_t ins[16];
   unsigned n, k;
   require((n = g5p_usc_iterator(ins, 0, 1, 4, true, false)) != 0, "varying FS: iterator");
   require((k = g5p_usc_wait(ins + n, 0)) != 0, "varying FS: wait");
   n += k;
   require((k = g5p_usc_emit(ins + n, 2, 0)) != 0, "varying FS: emit");
   n += k;
   n += g5p_usc_special(ins + n, true);
   for (unsigned i = 0; i < n; i++) put(out, ins[i]);
   return true;
}

int pco_g5p_compile_shape(nir_shader *nir, uint8_t **data, size_t *size, bool half_output,
                          unsigned *varyings)
{
   *varyings = 0;
   if(nir->info.stage!=MESA_SHADER_VERTEX && nir->info.stage!=MESA_SHADER_FRAGMENT)return 0;
   nir_intrinsic_instr *outputs[2]={NULL,NULL};unsigned n=0;
   nir_foreach_block(block,nir_shader_get_entrypoint(nir)) {
      if(block->cf_node.parent->type!=nir_cf_node_function)return 0;
      nir_foreach_instr(ins,block) {
         if(ins->type==nir_instr_type_load_const)continue;
         if(ins->type!=nir_instr_type_intrinsic)return 0;
         nir_intrinsic_instr *in=nir_instr_as_intrinsic(ins);
         if(in->intrinsic==nir_intrinsic_load_input && nir->info.stage==MESA_SHADER_VERTEX)continue;
         if((in->intrinsic==nir_intrinsic_load_interpolated_input || in->intrinsic==nir_intrinsic_load_barycentric_pixel) &&
            nir->info.stage==MESA_SHADER_FRAGMENT)continue;
         if(in->intrinsic!=nir_intrinsic_store_output || n==2)return 0;
         outputs[n++]=in;
      }
   }
   for(unsigned i=0;i<n;i++)
      if(outputs[i]->num_components!=4 || nir_intrinsic_component(outputs[i]) ||
         nir_intrinsic_write_mask(outputs[i])!=15 || !nir_src_is_const(outputs[i]->src[1]) ||
         nir_src_as_uint(outputs[i]->src[1]))return 0;
   bool vertex=nir->info.stage==MESA_SHADER_VERTEX;
   if(vertex) {
      /* Order outputs: POS then VAR0. */
      if(n==2 && nir_intrinsic_io_semantics(outputs[0]).location!=VARYING_SLOT_POS){
         nir_intrinsic_instr *t=outputs[0];outputs[0]=outputs[1];outputs[1]=t;}
      if(!n || nir_intrinsic_io_semantics(outputs[0]).location!=VARYING_SLOT_POS ||
         !whole_vec4_input(outputs[0]->src[0].ssa,VERT_ATTRIB_GENERIC0))return 0;
      if(n==2 && (nir_intrinsic_io_semantics(outputs[1]).location!=VARYING_SLOT_VAR0 ||
                  !whole_vec4_input(outputs[1]->src[0].ssa,VERT_ATTRIB_GENERIC1)))return 0;
      *varyings=n-1;
   } else {
      if(n!=1 || nir_intrinsic_io_semantics(outputs[0]).location!=FRAG_RESULT_DATA0)return 0;
      if(!nir_src_is_const(outputs[0]->src[0])) {
         if(!half_output || !smooth_varying(outputs[0]->src[0].ssa,VARYING_SLOT_VAR0))return 0;
         struct binary binary={0};
         if(setjmp(reject))return 0;
         encode_varying_fs(&binary);
         *data=malloc(binary.size);if(!*data)return 0;
         memcpy(*data,binary.data,binary.size);*size=binary.size;*varyings=1;
         fprintf(stderr,"G5P native NIR -> USC (vc varying FS): %u bytes\n",binary.size);
         return PCO_G5P_SHAPE_FS_VARYING;
      }
   }
   uint32_t values[4];
   if(!vertex) for(unsigned j=0;j<4;j++) values[j]=nir_src_as_const_value(outputs[0]->src[0])[j].u32;
   void *mem=ralloc_context(NULL);
   if(!mem)return 0;
   pco_ctx *ctx=pco_ctx_create(NULL,NULL,mem);
   pco_shader *shader=pco_shader_create(ctx,nir,mem);
   pco_func *func=pco_func_create(shader,PCO_FUNC_TYPE_ENTRYPOINT,0);
   pco_block *block=pco_block_create(func);block->cf_node.parent=&func->cf_node;
   list_addtail(&block->cf_node.link,&func->body);
   pco_builder b=pco_builder_create(func,pco_cursor_after_block(block));
   if(!vertex && half_output) {
      uint16_t half[4];
      for(unsigned j=0;j<4;j++) {
         float x;memcpy(&x,&values[j],4);half[j]=_mesa_float_to_half(x);
      }
      values[0]=half[0]|((uint32_t)half[1]<<16);
      values[1]=half[2]|((uint32_t)half[3]<<16);
   }
   unsigned words=vertex ? 4*n : half_output ? 2 : 4;
   for(unsigned j=0;j<words;j++) {
      if(vertex)pco_uvsw_write(&b,pco_ref_hwreg(j,PCO_REG_CLASS_TEMP),pco_ref_imm32(j));
      else pco_movi32(&b,pco_ref_hwreg(j,PCO_REG_CLASS_PIXOUT),pco_ref_imm32(values[j]));
   }
   pco_end(shader);
   struct binary binary={0};
   if(setjmp(reject)) {ralloc_free(mem);return 0;}
   encode(shader,&binary);
   *data=malloc(binary.size);
   if(!*data){ralloc_free(mem);return 0;}
   memcpy(*data,binary.data,binary.size);*size=binary.size;
   fprintf(stderr,"G5P native NIR -> PCO -> USC: %u bytes\n",binary.size);
   ralloc_free(mem);
   return vertex ? (n==2 ? PCO_G5P_SHAPE_VS_VARYING : PCO_G5P_SHAPE_VS_PASSTHROUGH) : PCO_G5P_SHAPE_FS_CONSTANT;
}

bool pco_g5p_compile_simple(nir_shader *nir, uint8_t **data, size_t *size, bool half_output)
{
   unsigned varyings;
   return pco_g5p_compile_shape(nir, data, size, half_output, &varyings) != 0;
}
