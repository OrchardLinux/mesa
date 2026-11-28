/* SPDX-License-Identifier: MIT */
/*
 * G3: compute for the Apple A7 GPU (S5L8960X, PowerVR G6430, BVNC 1.70.4.3).
 *
 * The A7's shader ISA is Rogue (Apple's "agx0"), the ISA upstream pco
 * encodes, with three deltas measured on a J71 (iPad Air) or read in Apple's
 * A7 programs (AGXMetalA7.bundle/ds.A7):
 *
 *  - a single upper source with a 3-bit bank is 2 bytes (pco: 3);
 *  - st.32 [vN:vN+1] stores vN+2 .. vN+1+burst, the registers right after the
 *    address pair; the encoded data field does not select the data (Apple
 *    encodes it as the address + 2, and so does this backend);
 *  - no 4-byte st form (IMG's st with the extension byte wrote nothing).
 *
 * This is a narrow backend, not a compiler: it takes a compute shader whose
 * body is a list of 32-bit SSBO stores, each with
 *
 *    byte offset = s * lid + t * wgid + c
 *    value       = a * lid + b * wgid + d   (or constants, up to 4 components)
 *
 * (lid the local invocation index along x, wgid the workgroup ID along x,
 * s and a small constants) and refuses every other shader by name. It emits
 * only group forms that ran on the A7 (imadd64 in Apple's A7 form, the 3-byte
 * st with MCU write-through, wdf, nop.end) through pco's own field encoders
 * (pco_isa.h), and a PDS program that only writes the unified store: PDS data
 * -> unified and PDS temps (the lane) -> unified DOUTWs. No coefficient
 * registers, no common store, no IF1 writes: those wedged the A7's firmware.
 *
 * The workgroup ID never reaches the GPU: every workgroup gets a CDM record of
 * its own, whose PDS data carries t * wgid + c and b * wgid + d, folded with
 * the buffer address on the CPU (g3_pds_data()).
 */
#ifndef G3_COMPUTE_H
#define G3_COMPUTE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct nir_shader;
struct spirv_to_nir_options;
struct nir_shader_compiler_options;

#define G3_MAX_STORES 4
#define G3_MAX_COMPONENTS 4
#define G3_MAX_USC_BYTES 8192
#define G3_MAX_BUFFERS 8
#define G3_MAX_POOL 16
#define G3_MAX_PDS_WRITES (4 * G3_MAX_STORES + 2 + G3_MAX_POOL / 2)
#define G3_MAX_PDS_CODE (G3_MAX_PDS_WRITES + 4)
/* DOUTU q0, a 64-bit pair per data write, a control word per write, 16-byte aligned */
#define G3_MAX_PDS_DATA (2 + 3 * G3_MAX_PDS_WRITES + 4)
/* workgroups per dispatch: one CDM record each in the job's first 4 KiB control stream block */
#define G3_MAX_WORKGROUPS 64

/* x = lid * lid + wgid * wg + c, modulo 2^32 */
struct g3_affine {
   uint32_t lid, wg, c;
};

enum g3_pds_src {
   G3_PDS_LANE,      /* PDS temps q0, lower 32 bits: the instance (lane) index */
   G3_PDS_ADDRESS,   /* 64 bits: buffer address + offset.wg * wgid + offset.c */
   G3_PDS_ADDEND,    /* 64 bits: {value.wg * wgid + value.c, 0} */
   G3_PDS_CONSTANTS, /* 64 bits: two of the store's constant components */
   G3_PDS_WGID,      /* 64 bits: {workgroup ID x, workgroup count x} (g3_pco.c) */
   G3_PDS_BUFFER,    /* 64 bits: buffer[store]'s address (g3_pco.c: store is the buffer index) */
   G3_PDS_POOL,      /* 64 bits: pool[2 * pair], pool[2 * pair + 1] (g3_pco.c's constants) */
   G3_PDS_SHMEM,     /* 64 bits: buffer[store] + shmem_size * wgid: the workgroup's threadgroup memory */
};

/* g3_program.buffers[] entry that is the dispatch's threadgroup memory, not a descriptor (g3_pco.c) */
#define G3_SHMEM_SET 0xff

/* One DOUTW of the kernel's PDS program into unified register ao */
struct g3_pds_write {
   enum g3_pds_src src;
   uint8_t store, pair, ao;
};

struct g3_store {
   uint8_t set, binding;
   uint8_t components;
   struct g3_affine offset;
   struct g3_affine value[G3_MAX_COMPONENTS];
};

struct g3_program {
   uint8_t code[G3_MAX_USC_BYTES];
   unsigned code_size;

   unsigned num_stores;
   struct g3_store stores[G3_MAX_STORES];

   unsigned num_writes;
   struct g3_pds_write writes[G3_MAX_PDS_WRITES];

   unsigned local_size;    /* x; y and z are 1 */
   unsigned unified_regs;  /* registers vi0..vi(n-1) the program reads or writes */
   bool uses_wgid;         /* a store depends on the workgroup ID */

   /* compiled by pco (g3_pco.c) rather than the narrow path; then buffer[i] of g3_pds_data() is buffers[i]'s */
   bool pco;
   unsigned usc_temps;     /* USC temporaries the program allocates */
   unsigned entry_offset;  /* the entry point's offset in code */
   unsigned num_pool;      /* 32-bit constants outside Rogue's table 0..31, from the unified store */
   uint32_t pool[G3_MAX_POOL];
   unsigned shmem_size;    /* threadgroup memory bytes per workgroup, in a buffer the dispatch allocates */
   unsigned num_buffers;
   struct {
      uint8_t set, binding;
   } buffers[G3_MAX_BUFFERS];

   char refusal[160];
};

/*
 * NIR (straight from spirv_to_nir) -> A7 USC. False with out->refusal set
 * for a shader outside the forms above. Lowers nir in place.
 */
bool g3_compile_compute(struct nir_shader *nir, struct g3_program *out);

/* The kernel's PDS code (Rogue DOUT layout, the A8's), returns its dwords */
unsigned g3_pds_code(const struct g3_program *p, uint32_t code[G3_MAX_PDS_CODE]);

/*
 * The PDS data of workgroup wgid: q0 the DOUTU source {usc_offset, 0x10}, then
 * the program's 64-bit sources, then a DOUTW control word per write.
 * buffer[i] is store i's buffer address. Returns its dwords (a multiple of 4).
 */
unsigned g3_pds_data(const struct g3_program *p, uint32_t usc_offset, const uint64_t *buffer,
                     uint32_t wgid, uint32_t num_wg, uint32_t data[G3_MAX_PDS_DATA]);

/*
 * NIR -> A7 USC through pco's pipeline with the A7's deltas (g3_pco.c), for
 * shaders the narrow path refuses. pco_ctx is the device's pco_ctx; nir comes
 * from spirv_to_nir with g3_pco_spirv_options()/g3_pco_nir_options().
 */
bool g3_compile_pco(void *pco_ctx, struct nir_shader *nir, struct g3_program *out);
const struct spirv_to_nir_options *g3_pco_spirv_options(void);
const struct nir_shader_compiler_options *g3_pco_nir_options(void);

/* KERNEL0 of the program's CDM kernel record for pds_dwords of PDS data */
uint32_t g3_cdm_kernel0(const struct g3_program *p, unsigned pds_dwords);
/* KERNEL8: the workgroup size and max_instances, in the A7's measured forms */
uint32_t g3_cdm_kernel8(const struct g3_program *p);

/*
 * Rogue nop.end with pco's padding, the A7's empty program (run on the J71
 * as a7-cdm "empty" and as the end-of-pass K3 task): 04 80 ee 00 f2 ff ff ff
 */
extern const uint8_t g3_usc_nop_end[8];

/* Re-encode the R6 programs that ran on the A7 and compare; false on a mismatch */
bool g3_selftest(char *why, unsigned size);

/* "lid*4 + wgid*128 + 0" etc., for logs */
void g3_describe(const struct g3_program *p, char *buf, unsigned size);

#ifdef __cplusplus
}
#endif

#endif
