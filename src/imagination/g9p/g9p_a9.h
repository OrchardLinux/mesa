/* SPDX-License-Identifier: MIT */
#ifndef G9P_A9_H
#define G9P_A9_H
/* Compute texture words the dispatch puts in the common store (g9p_a9_tex.h) */
#define G9P_A9_MAX_UPLOADS 10
enum g9p_a9_upload_kind { G9P_A9_UP_IMAGE = 1, G9P_A9_UP_SAMPLER, G9P_A9_UP_FETCH_SAMPLER, G9P_A9_UP_SIZE, G9P_A9_UP_LOD };
struct g9p_a9_upload {
   unsigned char kind, set, binding, element;
   unsigned char combined; /* a sampler in a combined image sampler: its words at +32 */
   unsigned char level;    /* a size block's level */
   unsigned char common, dwords;
   uint32_t value; /* literal common-store LOD */
};
/* g9p_shader::a9_bank1_regs: the kernel record's register field allocates bank 1 (g9p_cdm.h) */
#define G9P_A9_HAS_BANK1_REGS 1
/* A9 threadgroup word of the workgroup ID z (g9p_shader::a9_wgid_z: the kernel reads it). Words 0-1 hold x and y,
 * word 2 the software-barrier counter (g5p_barrier.h, k = 2), shared variables start at word 4 (G9P_SHARED_BASE).
 * The kernel PDS writes z here from temps q1's upper half, as AGXMetalA9's PDSEncoderGen1::EncodeComputePDSProgram
 * (0x207d1ea4c) does after x and y when the kernel uses it (pvr_arch_g5p_compute.c). */
#define G9P_A9_TG_WGID_Z 3
#define G9P_A9_FIELDS bool a9; unsigned a9_bank1_regs; bool a9_bodyless; bool a9_wgid_z; unsigned a9_local_dwords; bool a9_has_barriers; bool a9_device_fence; \
   unsigned a9_num_uploads; struct g9p_a9_upload a9_uploads[G9P_A9_MAX_UPLOADS]; char a9_refusal[192];
#define G9P_A9_END_PAD(ctx) do { if ((ctx)->out->a9 && (2*(ctx)->c->n+6)%64==0) G9P_EMIT((ctx)->c,0xc000); } while(0)
/* Keep main's G9P numbering; only the A9 backend uses NIR-order breaks. */
#define G9P_A9_EMPTY_BREAK_BEFORE(ctx, side) do { if (!(ctx)->out->a9 || !(side)) (ctx)->ip++; } while (0)
#define G9P_A9_EMPTY_BREAK_AFTER(ctx, side) do { if ((ctx)->out->a9 && (side)) (ctx)->ip++; } while (0)
#define G9P_A9_CTX_FIELDS unsigned a9_barriers_remaining;
#define G9P_A9_GENERATE_INIT a9_generate_init(ctx, impl)
#define G9P_A9_SET_BODYLESS(ctx, n) do { \
   if ((ctx)->out->a9 && (ctx)->stage == MESA_SHADER_COMPUTE) \
      (ctx)->out->a9_bodyless = !(n); \
} while (0)
const char *g9p_a9_last_refusal(void);
struct g9p_shader;
unsigned g9p_a9_pipeline_max_lanes(const struct g9p_shader *shader);
bool g9p_a9_compile_compute(nir_shader *, struct g9p_shader *, void *);
#endif
