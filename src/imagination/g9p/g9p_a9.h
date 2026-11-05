/* SPDX-License-Identifier: MIT */
#ifndef G9P_A9_H
#define G9P_A9_H
#ifdef PVR_APPLE_G5P
#define G9P_A9_FIELDS bool a9; bool a9_bodyless; unsigned a9_local_dwords; bool a9_has_barriers; bool a9_device_fence;
#define G9P_A9_END_PAD(ctx) do { if ((ctx)->out->a9 && (2*(ctx)->c->n+6)%64==0) G9P_EMIT((ctx)->c,0xc000); } while(0)
#define G9P_A9_CTX_FIELDS unsigned a9_barriers_remaining;
#define G9P_A9_GENERATE_INIT a9_generate_init(ctx, impl)
#define G9P_A9_SET_BODYLESS(ctx, n) do { \
   if ((ctx)->out->a9 && (ctx)->stage == MESA_SHADER_COMPUTE) \
      (ctx)->out->a9_bodyless = !(n); \
} while (0)
struct g9p_shader;
bool g9p_a9_compile_compute(nir_shader *, struct g9p_shader *, void *);
#else
#define G9P_A9_FIELDS
#define G9P_A9_CTX_FIELDS
#define G9P_A9_GENERATE_INIT ((void)0)
#define G9P_A9_SET_BODYLESS(ctx, n) ((void)0)
#define G9P_A9_END_PAD(ctx) ((void)0)
#endif
#endif
