/* SPDX-License-Identifier: MIT */
/*
 * NIR -> G9P (Apple A10 shader ISA) compiler for compute shaders. First
 * version: one basic block, integer ALU, local/workgroup IDs and storage
 * buffers; everything else is rejected with a message.
 */
#ifndef G9P_COMPILE_H
#define G9P_COMPILE_H

#include <stdbool.h>
#include <stdint.h>

#include "compiler/nir/nir.h"

#define G9P_MAX_BUFFERS 8

/* The set of the push-constant slot (its address is loaded like a buffer's) */
#define G9P_PUSH_CONSTANTS (~0u)

struct g9p_compute_shader {
   /* USC code, a multiple of 64 bytes, ending with the end word */
   uint16_t *code;
   unsigned code_size;

   /*
    * Storage buffers by common-store slot: buffer k's address is loaded
    * into common-store dwords 2k-2k+1, the threads per workgroup (both
    * halves) into dword 2 * num_buffers.
    */
   unsigned num_buffers;
   struct {
      unsigned set, binding;
   } buffers[G9P_MAX_BUFFERS];

   /* Registers per thread, mask stack included (G9P_CDM_KERNEL_REGS) */
   unsigned num_regs;

   /* The workgroup ID is read from threadgroup words 0-2 */
   bool uses_workgroup_id;
   unsigned local_size[3];
};

const struct spirv_to_nir_options *g9p_spirv_options(void);
const nir_shader_compiler_options *g9p_nir_options(void);

/* The common-store slot of a (set, binding), added if new; -1 when full */
int g9p_buffer_slot(struct g9p_compute_shader *out, unsigned set, unsigned binding);

/*
 * Lowers and compiles @nir (a compute shader fresh from SPIR-V) into @out;
 * @out->code is allocated on @mem_ctx. Returns false with a message on
 * stderr for anything the backend does not handle yet.
 */
bool g9p_compile_compute(nir_shader *nir, struct g9p_compute_shader *out,
                         void *mem_ctx);

#endif /* G9P_COMPILE_H */
