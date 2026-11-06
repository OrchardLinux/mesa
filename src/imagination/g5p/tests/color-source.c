/* SPDX-License-Identifier: MIT
 * Generated-by: LLM. Independent native witness field expectations.
 */
#include <assert.h>
#include <stdio.h>
#include "g5p_color_source.h"
int main(void) {
 struct g5p_color_source_abi p;uint8_t code[G5P_MRT_EOT_USC_MAX];
 assert(g5p_color_source_make(16,1,G5P_COLOR_SOURCE_UINT,&p));
 assert(p.output_words==2 && p.write_key==0x2122 && !p.word32);
 unsigned n=g5p_color_source_eot(code,&p);assert(n==66);
 const uint8_t narrow[]={0x1c,0x05,0,0x51,0,8};assert(!memcmp(code+12,narrow,6));
 assert(g5p_color_source_make(16,1,G5P_COLOR_SOURCE_SINT,&p));assert(p.write_key==0x3122);
 assert(g5p_color_source_make(16,1,G5P_COLOR_SOURCE_UNORM16,&p));assert(p.write_key==0x2122);
 assert(g5p_color_source_make(16,2,G5P_COLOR_SOURCE_UINT,&p));assert(p.write_key==0x42462 && p.output_words==2);
 assert(g5p_color_source_make(16,4,G5P_COLOR_SOURCE_UINT,&p));assert(p.write_key==0x3442642 && p.output_words==2);
 assert(g5p_color_source_make(32,1,G5P_COLOR_SOURCE_UINT,&p));assert(p.write_key==0x2422 && p.word32 && p.output_words==2);
 n=g5p_color_source_eot(code,&p);assert(n==66);
 const uint8_t word32[]={0x1c,0x15,0,0x51,0,8};assert(!memcmp(code+12,word32,6));
 assert(g5p_color_source_make(8,1,G5P_COLOR_SOURCE_UINT,&p));
 assert(p.output_words==2 && p.write_key==0x2002 && !p.word32);
 n=g5p_color_source_eot(code,&p);assert(n==66 && !memcmp(code+12,narrow,6));
 assert(g5p_color_source_make(8,1,G5P_COLOR_SOURCE_SINT,&p));assert(p.write_key==0x3002);
 assert(!g5p_color_source_make(8,2,G5P_COLOR_SOURCE_UINT,&p));
 assert(!g5p_color_source_make(8,1,G5P_COLOR_SOURCE_UNORM16,&p));
 assert(!g5p_color_source_make(16,3,G5P_COLOR_SOURCE_UINT,&p));
 assert(!g5p_color_source_make(32,1,G5P_COLOR_SOURCE_UNORM16,&p));
 puts("COLOR SOURCE ABI PASS native16/32-bit fields, counts, EOT and rejected unknowns");
}
