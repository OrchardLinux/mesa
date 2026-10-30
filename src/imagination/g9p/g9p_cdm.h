/* SPDX-License-Identifier: MIT */
/*
 * Compute data master (CDM) control-stream records and PDS instructions of
 * the Apple A10 GPU (applegpu_g9p), encoded from decoded fields. Header-only;
 * shared with the orchard gpu-001 test tools (tools/g9p-cdm.h).
 *
 * Field sources, so that other A-chip bring-ups can tell what is known
 * (sections of the orchard gpu-001 COMPUTE.md unless stated):
 *   - decoded + tested: the field's meaning was varied on hardware;
 *   - observed: the value is what iPadOS writes (captures), meaning open;
 *   - Rogue: the meaning follows PowerVR Rogue (Mesa), not tested on A10.
 *
 * Addresses in records are offsets from the PDS heap base (the device
 * context's PDS execution base); DOUTU's USC offsets are from the USC code
 * base. powervr's PDS and USC heaps are those bases on this driver.
 */
#ifndef G9P_CDM_H
#define G9P_CDM_H

#include <stdint.h>

/* --- PDS instructions ("Apple PDS DOUT encoding") ----------------------- */

/* DOUT type, bits 0-2 (decoded + tested: D/W/U; C observed in the fence) */
#define G9P_DOUTD		0u	/* DMA from memory into the USC common store */
#define G9P_DOUTW		1u	/* write a 32/64-bit source into USC registers/store */
#define G9P_DOUTU		2u	/* kick a USC task */
#define G9P_DOUTC		5u	/* fence: acknowledged by the CDM's fence wait */

/* Source space, bits 8-9 (decoded + tested) */
#define G9P_PDS_DATA		0u	/* the task's PDS data */
#define G9P_PDS_TEMPS		2u	/* PDS temps: q0 = local ID; workgroup ID in IF1 */

#define G9P_PDS_END		(1u << 26)	/* last instruction (decoded + tested) */
/*
 * Bit 27, Rogue's CC (condition code) bit. Observed: iPadOS sets it on its
 * kernel tasks' DOUTU (0xfc000002) and not on the empty kernel's
 * (0xf4000002); both forms run. Not tested further.
 */
#define G9P_PDS_CC		(1u << 27)

/*
 * DOUT: @type from 64-bit source register @src64 of @space; @ctrl is the
 * PDS data dword holding the control word (DOUTD/DOUTW destination and
 * size, Rogue layout: last bit 31, destination offset, BSIZE in bits 0-1).
 */
static inline uint32_t g9p_pds_dout(uint32_t type, uint32_t src64, uint32_t space,
				    uint32_t ctrl, uint32_t flags)
{
	return 0xf0000000u | flags | ctrl << 10 | space << 8 | src64 << 3 | type;
}

/* HALT (observed at the end of every captured program) */
static inline uint32_t g9p_pds_halt(void)
{
	return 0xd3000000u;
}

/*
 * Branch to instruction @target unless the task is the coefficient-sync task
 * (IF1, once per workgroup). Observed and tested only as the first
 * instruction, so whether @target is absolute or relative is open.
 */
static inline uint32_t g9p_pds_bra_not_if1(uint32_t target)
{
	return 0xc2c00000u | target;
}

/* --- CDM records (8 words; "CDM kernel record", "Group IDs work") -------- */

/* Word 0 (K0) */
#define G9P_CDM_FENCE		(1u << 0)	/* wait for the USC fence ack (tested) */
#define G9P_CDM_ANY_CLUSTER	(1u << 1)	/* Rogue USC_TARGET: 0 all, 1 any */
/* bits 2-7: PDS data size, 16-byte blocks minus one (decoded + tested) */
#define G9P_CDM_DATA_BLOCKS(n)	((uint32_t)((n) - 1) << 2)
/*
 * Bits 8 and 13, set by iPadOS on every kernel record (0x2103); meaning
 * open (observed). Rogue has PDS temp and USC unified sizes there.
 */
#define G9P_CDM_KERNEL_SIZES	0x2100u
/*
 * Bits 30-31: block type. 0 = task, 2 = terminate (end of stream),
 * 3 = fence block (observed in iPadOS's pass ending, carries the DOUTC
 * program; tested).
 */
#define G9P_CDM_FENCE_BLOCK	(3u << 30)
#define G9P_CDM_TERMINATE	(2u << 30)

/*
 * Word 1 (K1) low bits: flags on the 16-byte aligned data offset. Observed
 * values: 0x2 on DMA (constants) tasks and the barrier kernel, 0xa on
 * iPadOS's constants record and barrier kernel, 0x4 on the fence blocks and
 * the fenced kernel. Tested: 0x4 on the fence blocks makes the end of pass
 * wait for all earlier work ("Resolved: c401 was the hang").
 */
#define G9P_CDM_K1_DMA		0x2u
#define G9P_CDM_K1_FENCE	0x4u

/*
 * Word 6 of internal one-thread records (observed: iPadOS's internal
 * records and one-thread kernels); words 6-7 of a kernel record encode the
 * group size, see g9p_cdm_kernel().
 */
#define G9P_CDM_ONE_THREAD	0x40000000u

struct g9p_cdm_record {
	uint32_t w[8];
};

/* A one-thread task or block: no workgroups, internal word 6/7. */
static inline struct g9p_cdm_record g9p_cdm_task(uint32_t k0, uint32_t data, uint32_t k1_flags,
						 uint32_t code)
{
	struct g9p_cdm_record r = { { k0, data | k1_flags, code, 0, 0, 0,
				      G9P_CDM_ONE_THREAD, 0 } };

	return r;
}

/*
 * A DMA-only task loading constants into the USC common store (the buffer
 * binding; iPadOS's constants record). Word 6 = 0x40000001 as observed on
 * every captured DMA task (low bit: Rogue's common-store allocation, open).
 */
static inline struct g9p_cdm_record g9p_cdm_dma_task(uint32_t k0, uint32_t data, uint32_t code)
{
	struct g9p_cdm_record r = g9p_cdm_task(k0, data, G9P_CDM_K1_DMA, code);

	r.w[6] |= 1;
	return r;
}

/*
 * A kernel dispatch of @gx x @gy x @gz workgroups of @threads threads.
 * Words 3-5: workgroups minus one (decoded + tested). Word 6: (threads mod
 * 32) << 27 | threadgroup allocation (decoded + tested on our setup; iPadOS
 * writes 1 for 8 threads, which aliases groups here). Word 7: (threads - 1)
 * << 20 (decoded + tested).
 */
static inline struct g9p_cdm_record g9p_cdm_kernel(uint32_t k0, uint32_t data, uint32_t code,
						   uint32_t gx, uint32_t gy, uint32_t gz,
						   uint32_t threads, uint32_t tg_alloc)
{
	struct g9p_cdm_record r = { { k0, data, code, gx - 1, gy - 1, gz - 1,
				      (threads % 32) << 27 | tg_alloc, (threads - 1) << 20 } };

	return r;
}

/* --- Internal programs and the end of a compute pass --------------------- */

/*
 * The device-level programs the end of a pass runs. iPadOS builds them once
 * per device (AGXMetalA10 DevicePDSStateLoader::setup); here the caller
 * places them and passes their offsets.
 */
struct g9p_pass_end_programs {
	uint32_t kick_code;	/* PDS: g9p_pds_kick_program() */
	uint32_t kick_data;	/* PDS data: {empty kernel USC offset, 0x10} */
	uint32_t fence_code;	/* PDS: g9p_pds_fence_program() */
	uint32_t fence_data;	/* PDS data for the fence blocks (16 bytes) */
};

/* Kick a one-thread USC task (iPadOS's internal kernels): 4 words */
static inline void g9p_pds_kick_program(uint32_t *w)
{
	w[0] = g9p_pds_bra_not_if1(2);
	w[1] = g9p_pds_halt();
	w[2] = g9p_pds_dout(G9P_DOUTU, 0, G9P_PDS_DATA, 0, G9P_PDS_END | G9P_PDS_CC);
	w[3] = g9p_pds_halt();
}

/* The fence blocks' program, as captured on iPadOS: 4 words */
static inline void g9p_pds_fence_program(uint32_t *w)
{
	w[0] = g9p_pds_bra_not_if1(2);
	w[1] = g9p_pds_halt();
	w[2] = g9p_pds_dout(G9P_DOUTC, 0, G9P_PDS_DATA, 0, 0);
	w[3] = g9p_pds_halt();
}

/*
 * The kick program's data: USC offset of the kernel and 0x10 (observed in
 * every captured kernel task; Rogue's DOUTU temps field would make it 2
 * temp blocks).
 */
static inline void g9p_pds_kick_data(uint32_t *w, uint32_t usc_offset)
{
	w[0] = usc_offset;
	w[1] = 0x10;
}

/*
 * The empty USC kernel the end of a pass runs (driver shader 39, observed;
 * also standing in for driver shader 38, whose `c401` hangs on Linux):
 * a jump to the end word, which starts the next 64-byte block. 33 16-bit
 * words (ISA.md: `0100 01pp 0000`, `c000` padding, `8000`).
 */
static inline void g9p_usc_empty_kernel(uint16_t *w)
{
	unsigned int i;

	w[0] = 0x0100;
	w[1] = 0x0140;
	w[2] = 0x0000;
	for (i = 3; i < 32; i++)
		w[i] = 0xc000;
	w[32] = 0x8000;
}

/*
 * The end of a compute pass, iPadOS's barrier (AGXMetalA10
 * ComputeContext::encodeBarrier; COMPUTE.md "Resolved: c401 was the hang"):
 * empty kernel, fence block, fenced empty kernel, fence block, each with
 * its K1 flag, then the terminate. Without it the last workgroups are lost
 * and the GPU can wedge. Writes 33 words.
 */
static inline void g9p_cdm_end_pass(uint32_t *w, const struct g9p_pass_end_programs *p)
{
	const struct g9p_cdm_record rec[4] = {
		g9p_cdm_task(G9P_CDM_ANY_CLUSTER, p->kick_data, G9P_CDM_K1_DMA, p->kick_code),
		g9p_cdm_task(G9P_CDM_FENCE_BLOCK | G9P_CDM_ANY_CLUSTER | G9P_CDM_FENCE,
			     p->fence_data, G9P_CDM_K1_FENCE, p->fence_code),
		g9p_cdm_task(G9P_CDM_ANY_CLUSTER | G9P_CDM_FENCE, p->kick_data,
			     G9P_CDM_K1_FENCE, p->kick_code),
		g9p_cdm_task(G9P_CDM_FENCE_BLOCK | G9P_CDM_ANY_CLUSTER | G9P_CDM_FENCE,
			     p->fence_data, G9P_CDM_K1_FENCE, p->fence_code),
	};
	unsigned int i, k;

	for (k = 0; k < 4; k++)
		for (i = 0; i < 8; i++)
			w[8 * k + i] = rec[k].w[i];
	w[32] = G9P_CDM_TERMINATE;
}

#endif
