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

/*
 * The A8 (G4P): PDS DOUT in Rogue's field layout, src1 (the control word's
 * data dword) at bits 16-23 and src0 (the 64-bit source: space << 5 | index)
 * at bits 8-14, where the A9 and A10 use Apple's extended form
 * (g9p_pds_dout(): src1 at 10-17, src0 at 3-9). AGXMetalA8's compute PDS
 * encoder (iOS 12.4, 0x1a1453d44) builds DOUTW as 0xf0004001 | src1 << 16
 * and DOUTU as 0xfc000002 | src0 << 8; Metal's A8 programs decode the same
 * (DOUTW q0, c2 = 0xf0024001). Arguments as g9p_pds_dout().
 */
static inline uint32_t g5p_pds_dout_g4p(uint32_t type, uint32_t src64, uint32_t space,
                                        uint32_t ctrl, uint32_t flags)
{
   return 0xf0000000u | flags | ctrl << 16 | (space << 5 | src64) << 8 | type;
}

/* A DOUT for the chip: Rogue's layout on the A8 (g4p), Apple's extended one otherwise */
#define G5P_PDS_DOUT(g4p, ...) ((g4p) ? g5p_pds_dout_g4p(__VA_ARGS__) : g9p_pds_dout(__VA_ARGS__))

/* KERNEL0's PDS data size in 16-byte units: the A8 counts them directly (every captured A8 record: one unit
 * = 1, k_store_one's 0x2106, its binding DMA's two = 2, 0x00040008), G9P/A9 minus one */
#define G5P_CDM_DATA_UNITS(g4p, n) ((g4p) ? (uint32_t)(n) << 2 : G9P_CDM_DATA_BLOCKS(n))

/*
 * The A8's CDM KERNEL8 max_instances (bits 27-31; 32 wraps to 0), as
 * AGXMetalA8 computes it (iOS 12.4, 0x1a1428938): a workgroup of 32 or more
 * threads gets 32; a smaller one gets one workgroup per task when the kernel
 * asks for it (variant +0x79b: its compute metadata's field 30 at vtable
 * offset 0x40, which Apple's compiler sets, of 25 test kernels, only for the one
 * with a device atomic and not for the one with a barrier; Mesa passes "the
 * kernel does device atomics"), else 8 workgroups
 * (under 5 threads) or 32 / threads, at most (m + 319) / (2m - 1) with m the
 * threadgroup memory in 64-byte units; max_instances is workgroups * threads.
 * Apple writes 8 for a 1x1x1 kernel (0x40000000, OrchardLinux/gpu a8/M2).
 */
static inline unsigned g5p_cdm_max_instances_g4p(unsigned threads, bool one_wg_per_task,
                                                 unsigned shared_bytes)
{
   if (threads >= 32)
      return 0;
   unsigned wgs;
   if (one_wg_per_task) {
      wgs = 1;
   } else {
      wgs = threads < 5 ? 8 : 32 / threads;
      const unsigned m = (shared_bytes + 63) / 64;
      if (m) {
         const unsigned fit = (m + 0x13f) / (2 * m - 1);
         if (fit < wgs)
            wgs = fit;
      }
   }
   return (wgs * threads) & 31;
}
#endif
