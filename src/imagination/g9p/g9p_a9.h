/* SPDX-License-Identifier: MIT */
#ifndef G9P_A9_H
#define G9P_A9_H
#ifdef PVR_APPLE_G5P
#define G9P_A9_FIELDS bool a9; unsigned a9_local_dwords;
struct g9p_shader;
bool g9p_a9_compile_compute(nir_shader *, struct g9p_shader *, void *);
#else
#define G9P_A9_FIELDS
#endif
#endif
