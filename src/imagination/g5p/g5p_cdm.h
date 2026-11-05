/* SPDX-License-Identifier: MIT */
/* A9 Rogue CDMCTRL KERNEL0..5,KERNEL8 (csbgen/rogue/cdm.xml).
 * Seven words: unlike A10 there are no extra local-size words. */
#ifndef G5P_CDM_H
#define G5P_CDM_H
#include <stdint.h>
#include <stdbool.h>
struct g5p_cdm_record { uint32_t w[7]; };
/* Rogue allocates whole 32-instance tasks for multi-task workgroups.
 * The shader's logical local size and entry mask remain unpadded. */
static inline unsigned g5p_cdm_workgroup_threads(unsigned logical)
{
   return logical > 32 ? (logical + 31) & ~31u : logical;
}
static inline uint32_t g5p_cdm_sizes(unsigned common, unsigned regs,
                                    unsigned data_blocks, bool fence)
{
   return common << 18 | (2 * ((regs+7)/8)) << 12 |
          (regs ? 1u << 8 : 0) | (data_blocks-1) << 2 | (fence ? 3 : 0);
}
static inline struct g5p_cdm_record g5p_cdm_dma_task(uint32_t sizes, uint32_t data, uint32_t code, unsigned common)
{
   return (struct g5p_cdm_record){ .w = {sizes | common << 18,data | 2,code,0,0,0,8u << 27} };
}
static inline struct g5p_cdm_record g5p_cdm_kernel(uint32_t sizes, uint32_t data, uint32_t code,
     unsigned x, unsigned y, unsigned z, unsigned threads, unsigned common)
{
   return (struct g5p_cdm_record){ .w = {sizes | common << 18,data,code,x-1,y-1,z-1,(threads-1) << 18} };
}
static inline struct g5p_cdm_record g5p_cdm_task(uint32_t flags, uint32_t data, uint32_t kind, uint32_t code)
{
   return (struct g5p_cdm_record){ .w = {flags,data | kind,code,0,0,0,8u << 27} };
}
/* KERNEL0 indirect-present, KERNEL1/2, KERNEL6/7, KERNEL8.
 * The hardware reads three workgroup counts from the GPU address. */
struct g5p_cdm_indirect_record { uint32_t w[6]; };
static inline struct g5p_cdm_indirect_record
g5p_cdm_indirect(uint32_t sizes, uint32_t data, uint32_t code,
                  uint64_t address, unsigned threads, unsigned common)
{
   return (struct g5p_cdm_indirect_record){ .w = {
      sizes | common << 18 | (1u << 29), data, code,
      (uint32_t)(address >> 32), (uint32_t)address, (threads - 1) << 18,
   } };
}
#endif
