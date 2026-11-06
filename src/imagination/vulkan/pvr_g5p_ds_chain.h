/* SPDX-License-Identifier: MIT
 * Sampled depth mip chains: ordinary A9 TPU rectangles, planar stencil.
 * Generated-by: LLM. Pure bounded arithmetic, independently host-testable.
 */
#ifndef PVR_G5P_DS_CHAIN_H
#define PVR_G5P_DS_CHAIN_H
#include <stdbool.h>
#include <stdint.h>
struct pvr_g5p_ds_chain {
   uint64_t level[16], stencil_level[16];
   uint64_t layer, stencil_layer, stencil_base, size;
};
static inline uint64_t g5p_ds_align(uint64_t n, uint64_t a) { return (n+a-1)&~(a-1); }
static inline bool pvr_g5p_ds_chain_plan(unsigned w, unsigned h, unsigned levels,
      unsigned layers, unsigned cpp, bool stencil, struct pvr_g5p_ds_chain *p)
{
   if (!w || !h || w>16384 || h>16384 || !levels || levels>15 || !layers ||
       layers>16384 || (cpp!=2 && cpp!=4)) return false;
   unsigned pw=1,ph=1,full=1;
   while(pw<w)pw*=2;
   while(ph<h)ph*=2;
   for(unsigned n=pw>ph?pw:ph;n>1;n>>=1)full++;
   if(levels>full)return false;
   *p=(struct pvr_g5p_ds_chain){0};
   unsigned count=layers>1 && levels>1?full:levels;
   uint64_t texels=0;
   for(unsigned l=0;l<count;l++){
      if(l<levels){p->level[l]=texels*cpp;p->stencil_level[l]=texels;}
      unsigned x=pw>>l,y=ph>>l;
      texels+=(uint64_t)(x?x:1)*(y?y:1);
   }
   p->layer=texels*cpp;p->stencil_layer=texels;
   if(layers>1){
      p->layer=g5p_ds_align(p->layer,p->layer>=16384?16384:64);
      p->stencil_layer=g5p_ds_align(p->stencil_layer,p->stencil_layer>=16384?16384:64);
   }
   p->size=p->layer*layers;
   if(stencil){
      /* Keep a padded ZLS level-zero plane from overlapping stencil. */
      unsigned side=pw>ph?pw:ph;if(side<32)side=32;
      uint64_t zls=(uint64_t)side*side*cpp;
      p->stencil_base=g5p_ds_align(p->size>zls?p->size:zls,4096);
      p->size=p->stencil_base+p->stencil_layer*layers;
   }
   return true;
}
#endif
