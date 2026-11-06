/* SPDX-License-Identifier: MIT
 * Generated-by: LLM. Native witness-r16 source-layout ABI,2026-11-06.
 * Logical quantization/packing stays with callers. This describes the
 * physical PBE channels:16-bit lanes use EOT bit4 clear,32-bit lanes set.
 */
#ifndef G5P_COLOR_SOURCE_H
#define G5P_COLOR_SOURCE_H
#include <stdbool.h>
#include <stdint.h>
#include "g5p_mrt_programs.h"
enum g5p_color_source_class { G5P_COLOR_SOURCE_UINT, G5P_COLOR_SOURCE_SINT, G5P_COLOR_SOURCE_UNORM16 };
struct g5p_color_source_abi {
   unsigned channel_bits, channels, output_words;
   bool word32;
   uint32_t write_key; /* native linear low PBE q0; write selectors, not sample selectors */
};
static inline bool g5p_color_source_make(unsigned bits, unsigned channels,
                                        enum g5p_color_source_class type, struct g5p_color_source_abi *p)
{
   *p=(struct g5p_color_source_abi){0};
   if ((bits!=16 && bits!=32) || (channels!=1 && channels!=2 && channels!=4) ||
       type>G5P_COLOR_SOURCE_UNORM16 || (type==G5P_COLOR_SOURCE_UNORM16 && bits!=16))
      return false; /* eight-bit source layout awaits its own native witness */
   unsigned layout=bits==16?(channels==1?9:channels==2?35:50):(channels==1?33:channels==2?49:56);
   uint32_t selectors=channels>=2?1u<<18:0;
   if(channels==4)selectors|=(2u<<21)|(3u<<24);
   unsigned words=(bits*channels+31)/32;
   *p=(struct g5p_color_source_abi){bits,channels,words<2?2:words,bits==32,
      2u|(layout<<5)|((type==G5P_COLOR_SOURCE_SINT?3u:2u)<<12)|selectors};
   return true;
}
static inline unsigned g5p_color_source_eot(uint8_t *out,const struct g5p_color_source_abi *p)
{
   return g5p_mrt_eot_usc_mode(out,1,p->word32);
}
#endif
