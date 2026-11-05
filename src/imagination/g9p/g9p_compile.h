/* SPDX-License-Identifier: MIT */
/*
 * NIR -> G9P (Apple A10 shader ISA) compiler for compute, vertex and
 * fragment shaders: control flow, integer and float ALU, storage buffers,
 * push constants, vertex attributes from memory, vec4 varyings and one
 * colour output; everything else is rejected with a message.
 */
#ifndef G9P_COMPILE_H
#define G9P_COMPILE_H

#include <stdbool.h>
#include <stdint.h>

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_lower_blend.h"
#include "util/format/u_formats.h"

/*
 * Common-store buffer slots per shader; a vertex binding takes two (its
 * address and stride), and Zink gives each attribute its own binding
 */
#define G9P_MAX_BUFFERS 36 /* 16 vertex bindings x address and stride, and more */
#define G9P_MAX_ATTRIBS 16
#define G9P_MAX_VARYINGS 16
#define G9P_MAX_TEXTURES 16
/*
 * Texture state in the common store (Apple's layout for n textures, metal-tt):
 * texture t's state (8 dwords) at dword 8t, its sampler (2 dwords) at 8n + 2t,
 * from dword 0 in fragment shaders and after the buffers' address slots in
 * vertex shaders. In fragment shaders the buffers' contents follow: buffer k's
 * first G9P_FS_BUF_DWORDS dwords at g9p_fs_buf_base() + G9P_FS_BUF_DWORDS k,
 * which the shader reads as uniform operands (as Apple's fragment code does).
 */
#define G9P_VS_TEX_BASE (2 * G9P_MAX_BUFFERS)
#define G9P_FS_BUF_DWORDS 128

static inline unsigned
g9p_tex_state(unsigned base, unsigned t)
{
   return base + 8 * t;
}

static inline unsigned
g9p_tex_sampler(unsigned base, unsigned n, unsigned t)
{
   return base + 8 * n + 2 * t;
}

static inline unsigned
g9p_fs_buf_base(unsigned n)
{
   return n <= 1 ? 16 : (10 * n + 15) & ~15u;
}

/* The set of the push-constant slot (its address is loaded like a buffer's) */
#define G9P_PUSH_CONSTANTS (~0u)
/* The set of the dynamic blend constants (binding 0): 4 floats */
#define G9P_BLEND_CONSTANTS (~5u)
/* The set of the front face state: one word, ~0 for clockwise */
#define G9P_FRONT_FACE (~6u)
/* The set of vertex buffer slots (binding = the vertex buffer binding) */
#define G9P_VERTEX_BUFFERS (~1u)
/*
 * The set of the viewport buffer (binding 0): 8 floats, x/y/z scale and
 * offset (scale x, offset x, scale y, offset y, scale z, offset z), read by
 * vertex shaders that pass the fragment position on
 */
#define G9P_VIEWPORT (~2u)
/*
 * The set of vertex buffer stride slots (binding = the vertex buffer
 * binding): the stride as a 64-bit value, filled per draw (strides can be
 * dynamic state)
 */
#define G9P_VERTEX_STRIDES (~3u)
/*
 * gl_FragCoord comes in a varying (window x, y, z, 1/w) the vertex shader
 * writes after its own (g9p_frag_coord_slot()), iterated without
 * perspective correction: the position read's 16-bit results are not
 * decoded for f32 use yet
 */

/*
 * A vertex attribute from a vertex buffer: loaded as 32-bit words, and
 * formats with smaller channels (Qt Quick's 8-bit colours) unpacked in the
 * shader
 */
struct g9p_vertex_attrib {
   bool present;
   bool per_instance; /* indexed by the instance (r1), not the vertex (r0) */
   unsigned binding, offset;
   enum pipe_format format;
};

struct g9p_shader {
   mesa_shader_stage stage;

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
      /* fragment: how many dwords of it the shader reads (from its start) */
      unsigned dwords;
   } buffers[G9P_MAX_BUFFERS];

   /* Registers per thread, mask stack included (G9P_CDM_KERNEL_REGS) */
   unsigned num_regs;

   /* Compute: the workgroup ID is read from threadgroup words 0-2 */
   bool uses_workgroup_id;
   unsigned local_size[3];

   /*
    * Vertex: varyings written (VARn, each 4 components after the position);
    * fragment: varyings read and whether perspective-correct
    */
   unsigned num_varyings;
   bool varying_perspective[G9P_MAX_VARYINGS];
   /* Vertex: output dwords (4 + 4 x num_varyings) */
   unsigned num_outputs;

   /* Fragment: reads gl_FragCoord (the frag_coord_slot varying) */
   bool uses_frag_coord;
   /* Fragment: may discard (punch-through pass type) */
   bool uses_discard;
   bool uses_fbfetch;
   /*
    * Fragment: premultiplied source-over blended at the end with Apple's
    * half-precision sequence (g9p_blend_src_over()) instead of
    * nir_lower_blend's f32 code
    */
   bool native_blend;

   /* Fragment: the combined image samplers sampled, by common-store slot */
   unsigned num_textures;
   struct {
      unsigned set, binding, index;
   } textures[G9P_MAX_TEXTURES];
};

const struct spirv_to_nir_options *g9p_spirv_options(void);
const nir_shader_compiler_options *g9p_nir_options(void);

/* The common-store slot of a (set, binding), added if new; -1 when full */
int g9p_buffer_slot(struct g9p_shader *out, unsigned set, unsigned binding);

/*
 * Lowers and compiles @nir (a compute shader fresh from SPIR-V) into @out;
 * @out->code is allocated on @mem_ctx. Returns false with a message on
 * stderr for anything the backend does not handle yet.
 */
bool g9p_compile_compute(nir_shader *nir, struct g9p_shader *out, void *mem_ctx);

/* The varying slot after the vertex shader's own: gl_FragCoord's */
unsigned g9p_frag_coord_slot(const nir_shader *vs);

/*
 * Vertex shader: @attribs by attribute location (VERT_ATTRIB_GENERIC0 +
 * n); the vertex index arrives in r0. Vertex buffer b's address is loaded
 * into the common store as buffer (G9P_VERTEX_BUFFERS, b). It writes
 * gl_FragCoord's varying at @frag_coord_slot, or not with -1.
 */
bool g9p_compile_vertex(nir_shader *nir, const struct g9p_vertex_attrib *attribs,
                        int frag_coord_slot, struct g9p_shader *out, bool point_size,
                        void *mem_ctx);

/*
 * Fragment shader: @blend, if not NULL, is done in the shader (Apple's way:
 * nir_lower_blend, reading the tile's colour); out->uses_fbfetch tells the
 * pipeline to mark its draws translucent.
 */
bool g9p_compile_fragment(nir_shader *nir, unsigned frag_coord_slot,
                          const nir_lower_blend_options *blend, struct g9p_shader *out,
                          void *mem_ctx);

#endif /* G9P_COMPILE_H */
