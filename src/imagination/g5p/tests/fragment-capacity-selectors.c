/* SPDX-License-Identifier: MIT
 * Generated-by: LLM
 * Native applegpu_g5p implicit f16 descriptor-selector witnesses.
 * Compile with ../g5p_usc.c and -I..; host-only, does not qualify pixels.
 */
#include "g5p_usc.h"
#include <assert.h>
#include <string.h>
int main(void){unsigned char out[32];
 {struct g5p_tex t={.dst_half=48,.image_qword=0,.sampler_qword=64,.coord_dword=0,.coord_control=false,.class_variant=0,.dim=G5P_TEX_2D};unsigned char expected[]={26,245,48,1,128,1}; assert(g5p_usc_sample(out,&t)==sizeof(expected));assert(!memcmp(out+3,expected+3,2));}
 {struct g5p_tex t={.dst_half=58,.image_qword=4,.sampler_qword=65,.coord_dword=0,.coord_control=false,.class_variant=0,.dim=G5P_TEX_2D};unsigned char expected[]={26,245,58,9,130,1}; assert(g5p_usc_sample(out,&t)==sizeof(expected));assert(!memcmp(out+3,expected+3,2));}
 {struct g5p_tex t={.dst_half=40,.image_qword=8,.sampler_qword=66,.coord_dword=0,.coord_control=false,.class_variant=0,.dim=G5P_TEX_2D};unsigned char expected[]={26,245,40,17,132,1}; assert(g5p_usc_sample(out,&t)==sizeof(expected));assert(!memcmp(out+3,expected+3,2));}
 {struct g5p_tex t={.dst_half=30,.image_qword=12,.sampler_qword=67,.coord_dword=0,.coord_control=false,.class_variant=0,.dim=G5P_TEX_2D};unsigned char expected[]={26,245,30,25,134,1}; assert(g5p_usc_sample(out,&t)==sizeof(expected));assert(!memcmp(out+3,expected+3,2));}
 {struct g5p_tex t={.dst_half=22,.image_qword=16,.sampler_qword=68,.coord_dword=0,.coord_control=false,.class_variant=0,.dim=G5P_TEX_2D};unsigned char expected[]={26,245,22,33,136,1}; assert(g5p_usc_sample(out,&t)==sizeof(expected));assert(!memcmp(out+3,expected+3,2));}
 {struct g5p_tex t={.dst_half=14,.image_qword=20,.sampler_qword=69,.coord_dword=0,.coord_control=false,.class_variant=0,.dim=G5P_TEX_2D};unsigned char expected[]={26,245,14,41,138,1}; assert(g5p_usc_sample(out,&t)==sizeof(expected));assert(!memcmp(out+3,expected+3,2));}
 {struct g5p_tex t={.dst_half=6,.image_qword=24,.sampler_qword=70,.coord_dword=0,.coord_control=false,.class_variant=0,.dim=G5P_TEX_2D};unsigned char expected[]={26,245,6,49,140,1}; assert(g5p_usc_sample(out,&t)==sizeof(expected));assert(!memcmp(out+3,expected+3,2));}
 {struct g5p_tex t={.dst_half=62,.image_qword=28,.sampler_qword=71,.coord_dword=0,.coord_control=false,.class_variant=0,.dim=G5P_TEX_2D};unsigned char expected[]={26,245,62,57,142,0}; assert(g5p_usc_sample(out,&t)==sizeof(expected));assert(!memcmp(out+3,expected+3,2));}
 {struct g5p_tex t={.dst_half=54,.image_qword=32,.sampler_qword=72,.coord_dword=0,.coord_control=false,.class_variant=1,.dim=G5P_TEX_2D};unsigned char expected[]={27,245,54,65,144,0}; assert(g5p_usc_sample(out,&t)==sizeof(expected));assert(!memcmp(out+3,expected+3,2));}
 {struct g5p_tex t={.dst_half=46,.image_qword=36,.sampler_qword=73,.coord_dword=0,.coord_control=false,.class_variant=1,.dim=G5P_TEX_2D};unsigned char expected[]={27,245,46,73,146,0}; assert(g5p_usc_sample(out,&t)==sizeof(expected));assert(!memcmp(out+3,expected+3,2));}
 {struct g5p_tex t={.dst_half=38,.image_qword=40,.sampler_qword=74,.coord_dword=0,.coord_control=false,.class_variant=1,.dim=G5P_TEX_2D};unsigned char expected[]={27,245,38,81,148,0}; assert(g5p_usc_sample(out,&t)==sizeof(expected));assert(!memcmp(out+3,expected+3,2));}
 {struct g5p_tex t={.dst_half=30,.image_qword=44,.sampler_qword=75,.coord_dword=0,.coord_control=false,.class_variant=1,.dim=G5P_TEX_2D};unsigned char expected[]={27,245,30,89,150,0}; assert(g5p_usc_sample(out,&t)==sizeof(expected));assert(!memcmp(out+3,expected+3,2));}
 {struct g5p_tex t={.dst_half=24,.image_qword=48,.sampler_qword=76,.coord_dword=0,.coord_control=false,.class_variant=0,.dim=G5P_TEX_2D};unsigned char expected[]={26,245,24,97,152,0}; assert(g5p_usc_sample(out,&t)==sizeof(expected));assert(!memcmp(out+3,expected+3,2));}
 {struct g5p_tex t={.dst_half=16,.image_qword=52,.sampler_qword=77,.coord_dword=0,.coord_control=false,.class_variant=0,.dim=G5P_TEX_2D};unsigned char expected[]={26,245,16,105,154,0}; assert(g5p_usc_sample(out,&t)==sizeof(expected));assert(!memcmp(out+3,expected+3,2));}
 {struct g5p_tex t={.dst_half=8,.image_qword=56,.sampler_qword=78,.coord_dword=0,.coord_control=false,.class_variant=0,.dim=G5P_TEX_2D};unsigned char expected[]={26,245,8,113,156,0}; assert(g5p_usc_sample(out,&t)==sizeof(expected));assert(!memcmp(out+3,expected+3,2));}
 {struct g5p_tex t={.dst_half=0,.image_qword=60,.sampler_qword=79,.coord_dword=0,.coord_control=true,.class_variant=0,.dim=G5P_TEX_2D};unsigned char expected[]={26,245,0,121,158,128}; assert(g5p_usc_sample(out,&t)==sizeof(expected));assert(!memcmp(out+3,expected+3,2));}
return 0;}
