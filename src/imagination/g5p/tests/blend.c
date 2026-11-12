/* SPDX-License-Identifier: MIT
 * Generated-by: LLM
 * Exhaustive RGB/alpha equation construction and independent float oracle.
 * Include the implementation to check its scheduled physical-half dataflow,
 * not a second implementation of its algebraic lowering.
 */
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#define G5P_BLEND_TEST
#include "../g5p_blend.c"

static struct g5p_blend_key current;
static unsigned tested, evaluated, fallback;
static double source[4], destination[4], constants[4], secondary[4];
static unsigned max_bytes,max_half;
static unsigned current_mask = 0xf;
static unsigned current_target, current_targets = 1;
#ifdef G5P_BLEND_BASELINE
extern unsigned g5p_blend_baseline(uint8_t *,unsigned,const struct g5p_blend_key *,unsigned,
                                 struct g5p_blend_info *,const char **);
static unsigned old_identical;
#endif
static double factor_ref(unsigned f,unsigned l)
{
   switch(f) {
   case 0:return 0;case 1:return 1;case 2:return source[l];case 3:return 1-source[l];
   case 4:return destination[l];case 5:return 1-destination[l];case 6:return source[3];
   case 7:return 1-source[3];case 8:return destination[3];case 9:return 1-destination[3];
   case 10:return constants[l];case 11:return 1-constants[l];case 12:return constants[3];
   case 13:return 1-constants[3];case 14:return l==3?1:fmin(source[3],1-destination[3]);
   case 15:return secondary[l];case 16:return 1-secondary[l];
   case 17:return secondary[3];case 18:return 1-secondary[3];
   default:abort();
   }
}
static double clamp(double x){return fmin(1,fmax(0,x));}
static double equation_ref(unsigned l)
{
   if (!(current_mask & (1u << l)))
      return destination[l];
   unsigned op=l<3?current.op_rgb:current.op_a;
   if(op==3)return fmin(source[l],destination[l]);
   if(op==4)return fmax(source[l],destination[l]);
   double s=source[l]*factor_ref(l<3?current.src_rgb:current.src_a,l);
   double d=destination[l]*factor_ref(l<3?current.dst_rgb:current.dst_a,l);
   return clamp(op==0?s+d:op==1?s-d:d-s);
}
static void g5p_blend_test_program(const struct bctx *c,const struct g5p_blend_info *info)
{
   fallback+=c->general;
   /* The cases include negative/above-one constants after dynamic-state
    * clamping, distinct RGB/alpha, exact0/1 and non-symmetric source/dest.
    */
   for(unsigned sample=0;sample<3;sample++) {
      double h[64];
      for(unsigned i=0;i<64;i++)h[i]=NAN;
      for(unsigned t=0;t<current_targets;t++)for(unsigned l=0;l<4;l++)
         h[4*t+l]=0.125*(l+1);
      for(unsigned l=0;l<4;l++) {
         source[l]=sample==0?(double)(l+1)/8:sample==1?(double)(7-l)/8:(l==3?0:1);
         destination[l]=sample==0?(double)(7-l)/16:sample==1?(double)(l+2)/16:(l==3?1:0);
         constants[l]=sample==0?(double)(2*l+1)/8:sample==1?clamp((double)l-1):0.375;
         secondary[l]=sample==0?(double)(5-l)/8:sample==1?(double)(l+1)/8:(l==3?0.5:1);
         if(current_targets==1)h[8+l]=secondary[l];
         h[4*current_target+l]=info->zero_lanes&(1u<<l)?0:source[l];
      }
      for(unsigned v=0;v<c->nn;v++)if(c->n[v].op==B_RAW)h[c->n[v].half]=destination[c->n[v].lane];
      for(unsigned i=0;i<c->ni;i++) {
         const struct bnode *n=&c->n[c->ins[i].node];
         double s[4]={0};
         for(unsigned j=0;j<4;j++)if(n->s[j]>=0)s[j]=h[c->n[n->s[j]].half];
         double value=NAN;
         switch(n->op) {
         case B_TC:value=n->stepped&&!c->ins[i].step?h[n->half]:s[0];break;
         case B_ONEM:case B_ONEMC:value=1-s[1];break;
         case B_MOVC:value=constants[n->lane-c->const_base];break;
         case B_MUL:value=s[0]*s[1];break;
         case B_SQUARE:value=s[0]*s[0];break;
         case B_MULC:value=constants[n->lane-c->const_base]*s[1];break;
         case B_SUBC:value=constants[n->lane-c->const_base]-s[1];break;
         case B_ADD:value=s[0]+(n->neg?-s[1]:s[1]);break;
         case B_FMA:value=s[0]*s[1]+s[2];break;
         case B_LERP:value=s[0]+s[1]*((n->neg?-s[2]:s[2])-s[0]);break;
         case B_DOT2:value=s[0]*s[1]-s[2]*s[3];break;
         case B_MINMAX:value=n->max?fmax(s[0],s[1]):fmin(s[0],s[1]);break;
         case B_FIXUP:value=s[0];break;
         case B_COPY:value=s[1];break;case B_NEG:value=-s[1];break;case B_ZERO:value=0;break;
         default:abort();
         }
         h[n->half]=value;
      }
      for(unsigned l=0;l<4;l++) {
         double expected=equation_ref(l),got=clamp(h[4*current_target+l]);
         if(!isfinite(h[4*current_target+l]) || fabs(got-expected)>1e-12) {
            fprintf(stderr,"BLEND MATH FAIL key=%u,%u,%u,%u,%u,%u sample=%u lane=%u got=%.17g expected=%.17g general=%u\n",
             current.src_rgb,current.dst_rgb,current.op_rgb,current.src_a,current.dst_a,current.op_a,sample,l,got,expected,c->general);
            exit(1);
         }
      }
      for(unsigned t=0;t<current_targets;t++)if(t!=current_target)
         for(unsigned l=0;l<4;l++)assert(h[4*t+l]==0.125*(l+1));
      evaluated++;
   }
}
int main(int argc,char **argv)
{
   bool maskdump=argc==3&&!strcmp(argv[1],"--dump-masks");
   bool dump=argc==3&&(!strcmp(argv[1],"--dump")||maskdump);
   FILE *file=dump?fopen(argv[2],"wb"):NULL;
   if(dump&&!file)return 2;
   bool dual=argc==2&&!strcmp(argv[1],"--dual");
   bool mrt=argc==2&&!strcmp(argv[1],"--mrt");
   unsigned stride=(dump||mrt)?137:1;
   if(mrt) {
      uint8_t bytes[8];
      assert(g5p_usc_tile_read_component(bytes,0,2,4)==6);
      const uint8_t witness[]={0x1c,0x21,0,0x29,4,0};
      assert(!memcmp(bytes,witness,6)); /* mrtm-blend1 native first instruction */
      assert(g5p_usc_tile_read_component(bytes,6,3,6)==6);
      const uint8_t witness3[]={0x1c,0x31,6,0x29,6,0};
      assert(!memcmp(bytes,witness3,6)); /* mrtm-3bm native first instruction */
   }
   bool masks=mrt||maskdump||(argc==2&&!strcmp(argv[1],"--masks"));
   for(current_targets=mrt?2:1;current_targets<=(mrt?4:1);current_targets++)
   for(current_target=0;current_target<current_targets;current_target++)
   for(unsigned mask=masks?0:15;mask<16;mask++) {
   current_mask=mask;
   unsigned factors=dual?19:15, equations=5*factors*factors;
   for(unsigned rgb=0;rgb<equations;rgb+=mrt?17:1)for(unsigned alpha=0;alpha<equations;alpha+=stride) {
      current=(struct g5p_blend_key){rgb%factors,(rgb/factors)%factors,rgb/(factors*factors),alpha%factors,(alpha/factors)%factors,alpha/(factors*factors)};
      if(dual && current.src_rgb<15 && current.dst_rgb<15 && current.src_a<15 && current.dst_a<15)continue;
      uint8_t code[2048];struct g5p_blend_info info;const char *err;
      unsigned n=mrt?g5p_blend_target(code,sizeof(code),&current,0,current_mask,current_target,current_targets,&info,&err)
                    :g5p_blend_masked(code,sizeof(code),&current,0,current_mask,&info,&err);
      if(err) {
         fprintf(stderr,"BLEND BUILD FAIL key=%u,%u,%u,%u,%u,%u: %s\n",current.src_rgb,current.dst_rgb,current.op_rgb,current.src_a,current.dst_a,current.op_a,err);return 1;
      }
#ifdef G5P_BLEND_BASELINE
      if (current_mask==0xf) {
      uint8_t old[512];struct g5p_blend_info oi;const char *oe;
      unsigned on=g5p_blend_baseline(old,sizeof(old),&current,0,&oi,&oe);
      if(!oe) {
         if(on!=n || memcmp(old,code,n) || oi.max_half!=info.max_half || oi.reads_tile!=info.reads_tile ||
            oi.uses_constants!=info.uses_constants || oi.zero_lanes!=info.zero_lanes) {
            fprintf(stderr,"BLEND measured byte identity FAIL\n");return 1;
         }
         old_identical++;
      }
      }
#endif
      if(n>max_bytes)max_bytes=n;
      if(info.max_half>max_half)max_half=info.max_half;
      if(file) {
         if(maskdump)fprintf(file,"%u ",current_mask);
         fprintf(file,"%u %u %u %u %u %u %u %u ",current.src_rgb,current.dst_rgb,current.op_rgb,current.src_a,current.dst_a,current.op_a,info.zero_lanes,n);
         for(unsigned i=0;i<n;i++)fprintf(file,"%02x",code[i]);
         fputc('\n',file);
      }
      tested++;
   }
   }
   if(file)fclose(file);
   printf("BLEND UNIT PASS keys=%u float_models=%u canonical=%u max_bytes=%u max_half=%u\n",tested,evaluated,fallback,max_bytes,max_half);
#ifdef G5P_BLEND_BASELINE
   printf("BLEND BASELINE PASS old_buildable=%u byte_identical=%u\n",old_identical,old_identical);
#endif
   return 0;
}
