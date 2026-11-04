/* SPDX-License-Identifier: MIT */
/*
 * vertex-input: a hardware test of vertex input, for any Vulkan driver
 * (written for the PowerVR driver's Apple backends: G5P on the A9, G9P on the
 * A10). Plain Vulkan 1.1 (VK_KHR_index_type_uint8 and the zero vertex
 * attribute divisor when the device has them) and no driver internals.
 *
 * Every rung renders into a 64x64 linear B8G8R8A8_UNORM target cleared to
 * CLEAR, draws pixel-aligned quads (or lines) of constant colours, reads the
 * target back and compares every pixel with a CPU picture, exactly. Lines
 * leave their end pixels out of the comparison.
 *   list          a triangle list, position and colour in one binding (the baseline)
 *   bindings-2    position in binding 0, colour in binding 1
 *   bind-offsets  position at bind offset 16, colour at attribute offset 16 of a 32-byte stride
 *   indexed-16    16-bit indices
 *   indexed-32    32-bit indices, from index 6, with a vertex offset of 4
 *   indexed-8     8-bit indices (VK_KHR_index_type_uint8; skipped without it)
 *   instanced     4 instances of one quad, each moved and coloured by gl_InstanceIndex
 *   instanced-indexed  the same, indexed, with a vertex offset
 *   vertex-index  gl_VertexIndex picks each quad's blue (a plain draw from vertex 8)
 *   vertex-index-indexed  the same through indices and a vertex offset
 *   attribs-16    16 attributes over 3 bindings: a position and 15 floats summed in the shader
 *   strip         a triangle strip
 *   fan           triangle fans
 *   lines         a line list
 *   line-strip    a line strip
 *   int-imin int-imax int-umin int-umax int-isub int-ineg
 *                 one integer operation per vertex shader, an 8x8 grid of quads (quad index x instance index), each
 *                 channel one form of it (register pair, small literal, push constant, long literal) against the
 *                 same arithmetic on the CPU
 *   fmt-unorm8 fmt-bgra8 fmt-snorm8 fmt-uscaled8 fmt-sscaled8 fmt-unorm16 fmt-snorm16 fmt-uscaled16
 *   fmt-sscaled16 fmt-half fmt-offsets8 fmt-offsets16 fmt-int32
 *                 8 quads whose colour (or position) has the rung's vertex format, times a scale plus a bias, against
 *                 the format's conversion on the CPU (values a quarter step away from 8-bit rounding); a format
 *                 without VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT is skipped
 *
 *   inst-first    instances 2 and 3 of one quad (firstInstance 2), gl_InstanceIndex
 *   inst-first-indexed  instances 1..3, indexed with a vertex offset (firstInstance 1)
 *   inst-rate     a per-instance colour and offset (divisor 1), 4 instances from firstInstance 1
 *   inst-divisor0 the offset per instance, the colour divisor 0 (every instance the first's), firstInstance 2
 *                 (VK_KHR/EXT_vertex_attribute_divisor's zero divisor; skipped without it)
 *   restart-16 restart-32 restart-8
 *                 8 quads as one indexed triangle strip, cut by primitive restart (0xffff, 0xffffffff, 0xff)
 *
 * usage: vertex-input [--list] [--device N] [--no-verify] [RUNG...]
 * Output: one "VERTEX-INPUT <rung> PASS|FAIL|SKIP ..." line per rung, then
 * "VERTEX-INPUT RESULT PASS|FAIL <n> of <m>" (a SKIP counts as neither).
 * Exit status 0 on PASS. --no-verify runs and retires every rung without
 * comparing (a fake device node). Build: ./build.sh [OUT_DIR].
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "vertex-input-spirv.h"

#define W 64
#define H 64
#define MIN2(a, b) ((a) < (b) ? (a) : (b))
#define MAX2(a, b) ((a) > (b) ? (a) : (b))
#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))
#define MAX_QUADS 8
#define MAX_BINDINGS 3
#define MAX_ATTRIBS 16
#define MAX_FLOATS 1024

static const float CLEAR[4] = { 0.2f, 0.4f, 0.6f, 0.8f };

enum shader {
   SH_COLOR, SH_INSTANCE, SH_VERTEX_INDEX, SH_ATTRIB16,
   SH_INT_IMIN, SH_INT_IMAX, SH_INT_UMIN, SH_INT_UMAX, SH_INT_ISUB, SH_INT_INEG,
   SH_FMT, SH_FMT_INT, SH_INST_ATTR,
};

struct rung {
   const char *name;
   const char *what;
};

static const struct rung rungs[] = {
   { "list", "a triangle list, position and colour in one binding" },
   { "bindings-2", "position in binding 0, colour in binding 1" },
   { "bind-offsets", "bind offset 16 and attribute offset 16" },
   { "indexed-16", "16-bit indices" },
   { "indexed-32", "32-bit indices from index 6 with vertex offset 4" },
   { "indexed-8", "8-bit indices" },
   { "instanced", "4 instances, gl_InstanceIndex" },
   { "instanced-indexed", "4 instances, indexed, vertex offset 4" },
   { "vertex-index", "gl_VertexIndex, a draw from vertex 8" },
   { "vertex-index-indexed", "gl_VertexIndex through indices and vertex offset 8" },
   { "attribs-16", "16 attributes over 3 bindings" },
   { "strip", "a triangle strip" },
   { "fan", "triangle fans" },
   { "lines", "a line list" },
   { "line-strip", "a line strip" },
   { "int-imin", "vertex imin: register pair, literal 1, push constant, literal -2" },
   { "int-imax", "vertex imax: register pair, literal 1, push constant, literal -2" },
   { "int-umin", "vertex umin: register pair, literal 5, push constant, literal 0x50000" },
   { "int-umax", "vertex umax: register pair, literal 3, push constant, literal 0x50000" },
   { "int-isub", "vertex isub: register pair, 100 - x, push constant - x, x - push constant" },
   { "int-ineg", "vertex ineg and iabs" },
   { "fmt-unorm8", "colour R8G8B8A8_UNORM" },
   { "fmt-bgra8", "colour B8G8R8A8_UNORM" },
   { "fmt-snorm8", "colour R8G8B8A8_SNORM (-128 clamps to -1)" },
   { "fmt-uscaled8", "colour R8G8B8A8_USCALED" },
   { "fmt-sscaled8", "colour R8G8B8A8_SSCALED" },
   { "fmt-unorm16", "colour R16G16B16A16_UNORM" },
   { "fmt-snorm16", "colour R16G16_SNORM, blue and alpha the fill" },
   { "fmt-uscaled16", "colour R16G16B16A16_USCALED" },
   { "fmt-sscaled16", "colour R16G16B16A16_SSCALED" },
   { "fmt-half", "position R16G16_SFLOAT, colour R16G16B16A16_SFLOAT" },
   { "fmt-offsets8", "colour R8G8B8A8_UNORM at byte 9 (bytes 1..3 of a dword and 0 of the next)" },
   { "fmt-offsets16", "colour R16G16_UNORM at byte 14 (the high half of a dword and the low half of the next)" },
   { "fmt-int32", "R32G32_UINT and R32G32_SINT" },
   { "inst-first", "instances 2 and 3 (firstInstance 2), gl_InstanceIndex" },
   { "inst-first-indexed", "instances 1..3, indexed, vertex offset 4 (firstInstance 1)" },
   { "inst-rate", "per-instance colour and offset, divisor 1, firstInstance 1" },
   { "inst-divisor0", "per-instance offset, colour divisor 0, firstInstance 2" },
   { "restart-16", "one 16-bit indexed strip of 8 quads, cut by 0xffff" },
   { "restart-32", "one 32-bit indexed strip of 8 quads, cut by 0xffffffff" },
   { "restart-8", "one 8-bit indexed strip of 8 quads, cut by 0xff" },
};

/* the int-* rungs' push constants (shaders.glsl INTOP_VERT) */
static const int32_t INT_PC[4] = { -4, 1, 5, 3 };

/* A rung's geometry, vertex input and expected picture */
struct scene {
   enum shader shader;
   VkPrimitiveTopology topology;
   unsigned bindings, attribs;
   VkVertexInputBindingDescription vib[MAX_BINDINGS];
   VkVertexInputAttributeDescription via[MAX_ATTRIBS];
   float data[MAX_BINDINGS][MAX_FLOATS];
   unsigned data_floats[MAX_BINDINGS];
   VkDeviceSize bind_offset[MAX_BINDINGS];
   /* indices: 0 none, else bytes per index */
   unsigned index_size;
   uint32_t indices[64];
   unsigned index_count;
   /* the draws (each a vkCmdDraw or vkCmdDrawIndexed) */
   unsigned draws;
   struct {
      uint32_t count, instances, first;
      int32_t vertex_offset;
      uint32_t first_instance;
   } draw[4];
   /* vertex push constants (32 bytes), when push is set */
   bool push;
   int32_t pc[8];
   /* primitive restart enabled */
   bool restart;
   /* per-binding divisors other than 1 (VkPipelineVertexInputDivisorStateCreateInfoKHR) */
   unsigned ndivisors;
   VkVertexInputBindingDivisorDescriptionKHR divisor[MAX_BINDINGS];
   /* vertex formats the rung needs VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT for */
   VkFormat need[4];
   /* expected: RGBA per pixel, and pixels not compared */
   uint8_t want[H][W][4];
   bool skip[H][W];
};

static VkPhysicalDevice pd;
static VkDevice dev;
static VkQueue queue;
static uint32_t queue_family;
static bool verify = true, has_uint8, has_divisor0;
static const char *cur = "setup";

#define CHECK(x)                                                                                        \
   do {                                                                                                 \
      VkResult r_ = (x);                                                                                \
      if (r_ != VK_SUCCESS) {                                                                           \
         printf("VERTEX-INPUT %s FAIL: %s = %d (line %d)\n", cur, #x, r_, __LINE__);                    \
         return 1;                                                                                      \
      }                                                                                                 \
   } while (0)

static uint8_t
unorm8(double v)
{
   v = v < 0 ? 0 : v > 1 ? 1 : v;
   return (uint8_t)lround(v * 255.0);
}

/* NDC of a pixel edge */
static float
nx(int x)
{
   return (float)x / (W / 2) - 1.0f;
}

static float
ny(int y)
{
   return (float)y / (H / 2) - 1.0f;
}

static void
fill(struct scene *s, int x0, int y0, int x1, int y1, const float rgba[4])
{
   for (int y = y0; y < y1; y++)
      for (int x = x0; x < x1; x++)
         for (unsigned c = 0; c < 4; c++)
            s->want[y][x][c] = unorm8(rgba[c]);
}

static void
put(struct scene *s, unsigned b, const float *v, unsigned n)
{
   memcpy(&s->data[b][s->data_floats[b]], v, n * sizeof(float));
   s->data_floats[b] += n;
}

/* the colours of quad q */
static void
quad_color(unsigned q, float out[4])
{
   out[0] = (float)(30 + 60 * (q % 4)) / 255.0f;
   out[1] = (float)(200 - 20 * q) / 255.0f;
   out[2] = (float)(50 + 25 * q) / 255.0f;
   out[3] = (float)(255 - 10 * q) / 255.0f;
}

/* One binding of (x, y, r, g, b, a) */
static void
color_input(struct scene *s)
{
   s->bindings = 1;
   s->vib[0] = (VkVertexInputBindingDescription){ 0, 24, VK_VERTEX_INPUT_RATE_VERTEX };
   s->attribs = 2;
   s->via[0] = (VkVertexInputAttributeDescription){ 0, 0, VK_FORMAT_R32G32_SFLOAT, 0 };
   s->via[1] = (VkVertexInputAttributeDescription){ 1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 8 };
}

/* A quad's 4 corners (x0,y0) (x1,y0) (x0,y1) (x1,y1) with colour c, as (x, y, rgba) */
static void
quad_vertices(struct scene *s, int x0, int y0, int x1, int y1, const float c[4])
{
   const float v[4][6] = {
      { nx(x0), ny(y0), c[0], c[1], c[2], c[3] }, { nx(x1), ny(y0), c[0], c[1], c[2], c[3] },
      { nx(x0), ny(y1), c[0], c[1], c[2], c[3] }, { nx(x1), ny(y1), c[0], c[1], c[2], c[3] } };
   for (unsigned i = 0; i < 4; i++)
      put(s, 0, v[i], 6);
}

/* quads q of a 4x2 grid of 16x32 cells, inset by 2 pixels */
static void
grid_rect(unsigned q, int *x0, int *y0, int *x1, int *y1)
{
   *x0 = 16 * (q % 4) + 2;
   *x1 = *x0 + 12;
   *y0 = 32 * (q / 4) + 2;
   *y1 = *y0 + 28;
}

static void
add_draw(struct scene *s, uint32_t count, uint32_t instances, uint32_t first, int32_t vertex_offset)
{
   s->draw[s->draws].count = count;
   s->draw[s->draws].instances = instances;
   s->draw[s->draws].first = first;
   s->draw[s->draws].vertex_offset = vertex_offset;
   s->draws++;
}

static void
indexed_quads(struct scene *s, unsigned quads, unsigned size, unsigned first_index, int vertex_offset)
{
   color_input(s);
   s->index_size = size;
   /* vertex_offset dummy vertices first, then 4 per quad */
   for (int i = 0; i < vertex_offset; i++)
      put(s, 0, (const float[6]){ 0 }, 6);
   s->index_count = first_index;
   for (unsigned i = 0; i < first_index; i++)
      s->indices[i] = 0;
   for (unsigned q = 0; q < quads; q++) {
      int x0, y0, x1, y1;
      float c[4];
      grid_rect(q, &x0, &y0, &x1, &y1);
      quad_color(q, c);
      quad_vertices(s, x0, y0, x1, y1, c);
      fill(s, x0, y0, x1, y1, c);
      const uint32_t b = 4 * q;
      const uint32_t idx[6] = { b, b + 1, b + 2, b + 2, b + 1, b + 3 };
      memcpy(&s->indices[s->index_count], idx, sizeof(idx));
      s->index_count += 6;
   }
   add_draw(s, 6 * quads, 1, first_index, vertex_offset);
}

static void
hline(struct scene *s, int x0, int x1, int y, const float c[4])
{
   fill(s, x0, y, x1, y + 1, c);
   /* the end pixels are the rasterization rule's: not compared */
   for (int x = x0 - 1; x <= x0 + 1; x++)
      s->skip[y][x] = true;
   for (int x = x1 - 2; x <= x1; x++)
      s->skip[y][x] = true;
}

/* ---- the fmt-* rungs ---- */
enum conv { CV_UN8, CV_SN8, CV_U8, CV_S8, CV_UN16, CV_SN16, CV_U16, CV_S16, CV_F16, CV_F32, CV_U32, CV_S32 };

/* one attribute of a fmt-* rung: its format, offset, components, component bytes, conversion; its component c
 * comes from memory channel chan[c] and feeds colour channel out + c */
struct fmt_field {
   unsigned location;
   VkFormat format;
   unsigned offset, comps, bytes;
   enum conv conv;
   uint8_t chan[4];
   unsigned out;
};

static uint16_t
f32_to_f16(float f)
{
   uint32_t x;
   memcpy(&x, &f, 4);
   const uint32_t sign = (x >> 16) & 0x8000;
   int e = (int)((x >> 23) & 0xff) - 127 + 15;
   uint32_t m = x & 0x7fffff;
   if (e <= 0)
      return sign; /* the rungs' values are normal */
   uint32_t h = sign | (uint32_t)e << 10 | m >> 13;
   const uint32_t rest = m & 0x1fff;
   if (rest > 0x1000 || (rest == 0x1000 && (h & 1)))
      h++;
   return h;
}

static double
f16_to_double(uint16_t h)
{
   const int e = (h >> 10) & 0x1f;
   const double m = h & 0x3ff;
   const double v = e ? ldexp(1.0 + m / 1024.0, e - 15) : ldexp(m / 1024.0, -14);
   return h & 0x8000 ? -v : v;
}

/* the value a shader reads from raw x */
static double
conv_value(enum conv cv, int64_t x)
{
   switch (cv) {
   case CV_UN8: return x / 255.0;
   case CV_SN8: return MAX2(x / 127.0, -1.0);
   case CV_UN16: return x / 65535.0;
   case CV_SN16: return MAX2(x / 32767.0, -1.0);
   case CV_F16: return f16_to_double((uint16_t)x);
   case CV_F32: { float f; uint32_t u = (uint32_t)x; memcpy(&f, &u, 4); return f; }
   default: return (double)x;
   }
}

static void
conv_range(enum conv cv, int64_t *lo, int64_t *hi)
{
   switch (cv) {
   case CV_UN8: case CV_U8: *lo = 0; *hi = 255; break;
   case CV_SN8: case CV_S8: *lo = -128; *hi = 127; break;
   case CV_UN16: case CV_U16: *lo = 0; *hi = 65535; break;
   case CV_SN16: case CV_S16: *lo = -32768; *hi = 32767; break;
   case CV_U32: *lo = 0; *hi = 255; break;
   case CV_S32: *lo = -300; *hi = -45; break;
   default: *lo = 0; *hi = 255; break; /* F16: k / 255 rounded to f16 */
   }
}

/* raw x of an f16 rung value k (0..255) */
static int64_t
raw_of(enum conv cv, int64_t k)
{
   return cv == CV_F16 ? f32_to_f16((float)k / 255.0f) : k;
}

/* the 8-bit colour of value v times scale plus bias; *margin: its distance from an 8-bit rounding step */
static uint8_t
colour8(double v, double scale, double bias, double *margin)
{
   double e = (v * scale + bias) * 255.0;
   e = e < 0 ? 0 : e > 255 ? 255 : e;
   const double f = e - floor(e);
   *margin = e <= 0 || e >= 255 ? 1 : fabs(f - 0.5);
   return (uint8_t)lround(e);
}

static void
put_bytes(uint8_t *v, unsigned at, uint64_t x, unsigned bytes)
{
   for (unsigned i = 0; i < bytes; i++)
      v[at + i] = (uint8_t)(x >> (8 * i));
}

/* a fmt-* scene: 8 quads (grid_rect), 6 vertices each, of stride bytes: the position (R32G32_SFLOAT at 0, or
 * R16G16_SFLOAT at 0 with half_pos) and the fields; quad q's raw values are chosen per component from a
 * deterministic sequence, each moved on until its colour is a quarter step from 8-bit rounding: Vulkan lets
 * a colour round to either neighbour, and the A9 writes an 8-bit target through f16 rounded toward zero
 * (up to an eighth of a step below; device run of vtxin-6ba6f138bf3: 194.603 gave 194) */
static void
fmt_scene(struct scene *s, enum shader sh, unsigned stride, bool half_pos, const struct fmt_field *f, unsigned nf,
          const float scale[4], const float bias[4])
{
   s->shader = sh;
   s->push = true;
   memcpy(&s->pc[0], scale, 16);
   memcpy(&s->pc[4], bias, 16);
   s->bindings = 1;
   s->vib[0] = (VkVertexInputBindingDescription){ 0, stride, VK_VERTEX_INPUT_RATE_VERTEX };
   s->attribs = 1 + nf;
   s->via[0] = (VkVertexInputAttributeDescription){ 0, 0, half_pos ? VK_FORMAT_R16G16_SFLOAT : VK_FORMAT_R32G32_SFLOAT,
                                                    0 };
   if (half_pos)
      s->need[3] = VK_FORMAT_R16G16_SFLOAT;
   for (unsigned i = 0; i < nf; i++) {
      s->via[1 + i] = (VkVertexInputAttributeDescription){ f[i].location, 0, f[i].format, f[i].offset };
      s->need[i] = f[i].format;
   }
   for (unsigned q = 0; q < 8; q++) {
      int x0, y0, x1, y1;
      grid_rect(q, &x0, &y0, &x1, &y1);
      uint8_t vtx[32] = { 0 };
      float rgba[4] = { 0, 0, 0, 1 }; /* the shader's colour before scale and bias: the (0,0,0,1) fill */
      bool set[4] = { false };
      for (unsigned i = 0; i < nf; i++) {
         for (unsigned c = 0; c < f[i].comps; c++) {
            const unsigned o = f[i].out + c;
            int64_t lo, hi;
            conv_range(f[i].conv, &lo, &hi);
            /* the edges first (quads 0 and 1), then a spread */
            int64_t k = q == 0 ? lo : q == 1 ? hi : lo + (int64_t)((hi - lo) * ((q * 37 + o * 61 + i * 17) % 97) / 97);
            double v = 0, m = 0;
            for (unsigned tries = 0;; tries++) {
               v = conv_value(f[i].conv, raw_of(f[i].conv, k));
               colour8(v, scale[o], bias[o], &m);
               if (m >= 0.25 || tries > 1000)
                  break;
               k = k < hi ? k + 1 : lo;
            }
            rgba[o] = (float)v;
            set[o] = true;
            put_bytes(vtx, f[i].offset + f[i].bytes * f[i].chan[c], (uint64_t)raw_of(f[i].conv, k), f[i].bytes);
         }
      }
      uint8_t want[4];
      for (unsigned o = 0; o < 4; o++) {
         double m;
         want[o] = colour8(set[o] ? rgba[o] : (o == 3 ? 1.0 : 0.0), scale[o], bias[o], &m);
      }
      const int px[6] = { x0, x1, x0, x0, x1, x1 }, py[6] = { y0, y0, y1, y1, y0, y1 };
      for (unsigned k = 0; k < 6; k++) {
         if (half_pos) {
            put_bytes(vtx, 0, f32_to_f16(nx(px[k])), 2);
            put_bytes(vtx, 2, f32_to_f16(ny(py[k])), 2);
         } else {
            const float p[2] = { nx(px[k]), ny(py[k]) };
            memcpy(vtx, p, 8);
         }
         put(s, 0, (const float *)vtx, stride / 4);
      }
      for (int y = y0; y < y1; y++)
         for (int x = x0; x < x1; x++)
            memcpy(s->want[y][x], want, 4);
   }
   add_draw(s, 48, 1, 0, 0);
}

static bool
fmt_build(const char *name, struct scene *s)
{
   static const float one[4] = { 1, 1, 1, 1 }, zero[4] = { 0, 0, 0, 0 };
#define F(loc, fmt, off, n, b, cv, out, ...) { loc, VK_FORMAT_##fmt, off, n, b, cv, __VA_ARGS__, out }
   if (!strcmp(name, "fmt-unorm8")) {
      const struct fmt_field f[] = { F(1, R8G8B8A8_UNORM, 8, 4, 1, CV_UN8, 0, { 0, 1, 2, 3 }) };
      fmt_scene(s, SH_FMT, 16, false, f, 1, one, zero);
   } else if (!strcmp(name, "fmt-bgra8")) {
      const struct fmt_field f[] = { F(1, B8G8R8A8_UNORM, 8, 4, 1, CV_UN8, 0, { 2, 1, 0, 3 }) };
      fmt_scene(s, SH_FMT, 16, false, f, 1, one, zero);
   } else if (!strcmp(name, "fmt-snorm8")) {
      /* -1.0 (from -128 and -127) is 0.05 (12.75), -128 / 127 unclamped would be 11.85 */
      const struct fmt_field f[] = { F(1, R8G8B8A8_SNORM, 8, 4, 1, CV_SN8, 0, { 0, 1, 2, 3 }) };
      fmt_scene(s, SH_FMT, 16, false, f, 1, (const float[4]){ 0.45f, 0.45f, 0.45f, 0.45f },
                (const float[4]){ 0.5f, 0.5f, 0.5f, 0.5f });
   } else if (!strcmp(name, "fmt-uscaled8")) {
      const struct fmt_field f[] = { F(1, R8G8B8A8_USCALED, 8, 4, 1, CV_U8, 0, { 0, 1, 2, 3 }) };
      fmt_scene(s, SH_FMT, 16, false, f, 1, (const float[4]){ 1 / 255.0f, 1 / 255.0f, 1 / 255.0f, 1 / 255.0f }, zero);
   } else if (!strcmp(name, "fmt-sscaled8")) {
      const struct fmt_field f[] = { F(1, R8G8B8A8_SSCALED, 8, 4, 1, CV_S8, 0, { 0, 1, 2, 3 }) };
      fmt_scene(s, SH_FMT, 16, false, f, 1, (const float[4]){ 1 / 256.0f, 1 / 256.0f, 1 / 256.0f, 1 / 256.0f },
                (const float[4]){ 0.5f, 0.5f, 0.5f, 0.5f });
   } else if (!strcmp(name, "fmt-unorm16")) {
      const struct fmt_field f[] = { F(1, R16G16B16A16_UNORM, 8, 4, 2, CV_UN16, 0, { 0, 1, 2, 3 }) };
      fmt_scene(s, SH_FMT, 16, false, f, 1, one, zero);
   } else if (!strcmp(name, "fmt-snorm16")) {
      const struct fmt_field f[] = { F(1, R16G16_SNORM, 8, 2, 2, CV_SN16, 0, { 0, 1 }) };
      fmt_scene(s, SH_FMT, 16, false, f, 1, (const float[4]){ 0.5f, 0.5f, 1, 1 },
                (const float[4]){ 0.5f, 0.5f, 0.4f, 0 });
   } else if (!strcmp(name, "fmt-uscaled16")) {
      const struct fmt_field f[] = { F(1, R16G16B16A16_USCALED, 8, 4, 2, CV_U16, 0, { 0, 1, 2, 3 }) };
      const float k = 1 / 65535.0f;
      fmt_scene(s, SH_FMT, 16, false, f, 1, (const float[4]){ k, k, k, k }, zero);
   } else if (!strcmp(name, "fmt-sscaled16")) {
      const struct fmt_field f[] = { F(1, R16G16B16A16_SSCALED, 8, 4, 2, CV_S16, 0, { 0, 1, 2, 3 }) };
      const float k = 1 / 65536.0f;
      fmt_scene(s, SH_FMT, 16, false, f, 1, (const float[4]){ k, k, k, k }, (const float[4]){ 0.5f, 0.5f, 0.5f, 0.5f });
   } else if (!strcmp(name, "fmt-half")) {
      const struct fmt_field f[] = { F(1, R16G16B16A16_SFLOAT, 4, 4, 2, CV_F16, 0, { 0, 1, 2, 3 }) };
      fmt_scene(s, SH_FMT, 12, true, f, 1, one, zero);
   } else if (!strcmp(name, "fmt-offsets8")) {
      const struct fmt_field f[] = { F(1, R8G8B8A8_UNORM, 9, 4, 1, CV_UN8, 0, { 0, 1, 2, 3 }) };
      fmt_scene(s, SH_FMT, 16, false, f, 1, one, zero);
   } else if (!strcmp(name, "fmt-offsets16")) {
      const struct fmt_field f[] = { F(1, R16G16_UNORM, 14, 2, 2, CV_UN16, 0, { 0, 1 }) };
      fmt_scene(s, SH_FMT, 20, false, f, 1, one, zero);
   } else if (!strcmp(name, "fmt-int32")) {
      const struct fmt_field f[] = { F(1, R32G32_UINT, 8, 2, 4, CV_U32, 0, { 0, 1 }),
                                     F(2, R32G32_SINT, 16, 2, 4, CV_S32, 2, { 0, 1 }) };
      const float k = 1 / 255.0f;
      fmt_scene(s, SH_FMT_INT, 24, false, f, 2, (const float[4]){ k, k, k, k },
                (const float[4]){ 0, 0, 300 / 255.0f, 300 / 255.0f });
   } else {
      return false;
   }
#undef F
   return true;
}

/* Build rung r's scene. false: skipped (*why) */
static bool
build(const char *name, struct scene *s, const char **why)
{
   memset(s, 0, sizeof(*s));
   for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++)
         for (unsigned c = 0; c < 4; c++)
            s->want[y][x][c] = unorm8(CLEAR[c]);
   s->shader = SH_COLOR;
   s->topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

   if (!strcmp(name, "list")) {
      color_input(s);
      for (unsigned q = 0; q < 8; q++) {
         int x0, y0, x1, y1;
         float c[4];
         grid_rect(q, &x0, &y0, &x1, &y1);
         quad_color(q, c);
         const float v[6][6] = {
            { nx(x0), ny(y0), c[0], c[1], c[2], c[3] }, { nx(x1), ny(y0), c[0], c[1], c[2], c[3] },
            { nx(x0), ny(y1), c[0], c[1], c[2], c[3] }, { nx(x0), ny(y1), c[0], c[1], c[2], c[3] },
            { nx(x1), ny(y0), c[0], c[1], c[2], c[3] }, { nx(x1), ny(y1), c[0], c[1], c[2], c[3] } };
         for (unsigned i = 0; i < 6; i++)
            put(s, 0, v[i], 6);
         fill(s, x0, y0, x1, y1, c);
      }
      add_draw(s, 48, 1, 0, 0);
   } else if (!strcmp(name, "bindings-2") || !strcmp(name, "bind-offsets")) {
      /* bindings-2: position in binding 0 (stride 8), colour in binding 1 (stride 16).
       * bind-offsets: binding 0 bound 16 bytes in; binding 1's stride 32 with the colour at 16 */
      const bool offs = name[0] == 'b' && name[5] == 'o';
      s->bindings = 2;
      s->vib[0] = (VkVertexInputBindingDescription){ 0, 8, VK_VERTEX_INPUT_RATE_VERTEX };
      s->vib[1] = (VkVertexInputBindingDescription){ 1, offs ? 32 : 16, VK_VERTEX_INPUT_RATE_VERTEX };
      s->attribs = 2;
      s->via[0] = (VkVertexInputAttributeDescription){ 0, 0, VK_FORMAT_R32G32_SFLOAT, 0 };
      s->via[1] = (VkVertexInputAttributeDescription){ 1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, offs ? 16 : 0 };
      if (offs) {
         s->bind_offset[0] = 16;
         put(s, 0, (const float[4]){ 9, 9, 9, 9 }, 4);
      }
      for (unsigned q = 0; q < 8; q++) {
         int x0, y0, x1, y1;
         float c[4];
         grid_rect(q, &x0, &y0, &x1, &y1);
         quad_color(q, c);
         const int px[6] = { x0, x1, x0, x0, x1, x1 }, py[6] = { y0, y0, y1, y1, y0, y1 };
         for (unsigned i = 0; i < 6; i++) {
            put(s, 0, (const float[2]){ nx(px[i]), ny(py[i]) }, 2);
            if (offs)
               put(s, 1, (const float[4]){ -1, -1, -1, -1 }, 4);
            put(s, 1, c, 4);
         }
         fill(s, x0, y0, x1, y1, c);
      }
      add_draw(s, 48, 1, 0, 0);
   } else if (!strcmp(name, "indexed-16")) {
      indexed_quads(s, 8, 2, 0, 0);
   } else if (!strcmp(name, "indexed-32")) {
      indexed_quads(s, 8, 4, 6, 4);
   } else if (!strcmp(name, "indexed-8")) {
      if (!has_uint8) {
         *why = "no VK_KHR_index_type_uint8";
         return false;
      }
      indexed_quads(s, 8, 1, 0, 0);
   } else if (!strcmp(name, "instanced") || !strcmp(name, "instanced-indexed")) {
      const bool indexed = name[9] == '-';
      s->shader = SH_INSTANCE;
      color_input(s);
      const float c[4] = { 120 / 255.0f, 0, 70 / 255.0f, 1 };
      const int vo = indexed ? 4 : 0;
      for (int i = 0; i < vo; i++)
         put(s, 0, (const float[6]){ 0 }, 6);
      quad_vertices(s, 2, 10, 14, 54, c);
      for (unsigned i = 0; i < 4; i++) {
         const float ci[4] = { c[0], (float)(32 * (i + 1)) / 255.0f, c[2], c[3] };
         fill(s, 2 + 16 * i, 10, 14 + 16 * i, 54, ci);
      }
      if (indexed) {
         s->index_size = 2;
         const uint32_t idx[6] = { 0, 1, 2, 2, 1, 3 };
         memcpy(s->indices, idx, sizeof(idx));
         s->index_count = 6;
         add_draw(s, 6, 4, 0, vo);
      } else {
         s->topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
         add_draw(s, 4, 4, 0, 0);
      }
   } else if (!strcmp(name, "vertex-index") || !strcmp(name, "vertex-index-indexed")) {
      /* 8 vertices of padding, then 4 per quad: quad q's vertices are 8 + 4q.., blue 40 (2 + q) / 255 */
      const bool indexed = strlen(name) > 12;
      s->shader = SH_VERTEX_INDEX;
      color_input(s);
      for (unsigned i = 0; i < 8; i++)
         put(s, 0, (const float[6]){ 0 }, 6);
      for (unsigned q = 0; q < 6; q++) {
         int x0, y0, x1, y1;
         float c[4];
         grid_rect(q, &x0, &y0, &x1, &y1);
         quad_color(q, c);
         quad_vertices(s, x0, y0, x1, y1, c);
         c[2] = (float)(40 * (2 + q)) / 255.0f;
         fill(s, x0, y0, x1, y1, c);
         if (indexed) {
            const uint32_t b = 4 * q;
            const uint32_t idx[6] = { b, b + 1, b + 2, b + 2, b + 1, b + 3 };
            memcpy(&s->indices[s->index_count], idx, sizeof(idx));
            s->index_count += 6;
         }
      }
      if (indexed) {
         s->index_size = 4;
         add_draw(s, 36, 1, 0, 8);
      } else {
         /* a list of 6 vertices per quad would break the 4-per-quad index: one strip draw per quad */
         s->topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
         for (unsigned q = 0; q < 4; q++)
            add_draw(s, 4, 1, 8 + 4 * q, 0);
         /* the last two quads are not drawn */
         for (unsigned q = 4; q < 6; q++) {
            int x0, y0, x1, y1;
            grid_rect(q, &x0, &y0, &x1, &y1);
            fill(s, x0, y0, x1, y1, CLEAR);
         }
      }
   } else if (!strcmp(name, "attribs-16")) {
      /* binding 0: position, a1..a4 (stride 24); binding 1: a5..a10 (24); binding 2: a11..a15 (20).
       * a_k = (k + 4 q) / 1024: every sum is exact in float */
      s->shader = SH_ATTRIB16;
      s->bindings = 3;
      s->vib[0] = (VkVertexInputBindingDescription){ 0, 24, VK_VERTEX_INPUT_RATE_VERTEX };
      s->vib[1] = (VkVertexInputBindingDescription){ 1, 24, VK_VERTEX_INPUT_RATE_VERTEX };
      s->vib[2] = (VkVertexInputBindingDescription){ 2, 20, VK_VERTEX_INPUT_RATE_VERTEX };
      s->attribs = 16;
      s->via[0] = (VkVertexInputAttributeDescription){ 0, 0, VK_FORMAT_R32G32_SFLOAT, 0 };
      for (unsigned k = 1; k <= 15; k++) {
         const unsigned b = k <= 4 ? 0 : k <= 10 ? 1 : 2;
         const unsigned off = b == 0 ? 8 + 4 * (k - 1) : b == 1 ? 4 * (k - 5) : 4 * (k - 11);
         s->via[k] = (VkVertexInputAttributeDescription){ k, b, VK_FORMAT_R32_SFLOAT, off };
      }
      for (unsigned q = 0; q < 8; q++) {
         int x0, y0, x1, y1;
         float a[16], c[4] = { 0, 0, 0, 1 };
         grid_rect(q, &x0, &y0, &x1, &y1);
         for (unsigned k = 1; k <= 15; k++) {
            a[k] = (float)(k * 9 + 4 * q) / 1024.0f;
            c[(k - 1) / 5] += a[k];
         }
         const int px[6] = { x0, x1, x0, x0, x1, x1 }, py[6] = { y0, y0, y1, y1, y0, y1 };
         for (unsigned i = 0; i < 6; i++) {
            put(s, 0, (const float[6]){ nx(px[i]), ny(py[i]), a[1], a[2], a[3], a[4] }, 6);
            put(s, 1, &a[5], 6);
            put(s, 2, &a[11], 5);
         }
         fill(s, x0, y0, x1, y1, c);
      }
      add_draw(s, 48, 1, 0, 0);
   } else if (!strcmp(name, "inst-first") || !strcmp(name, "inst-first-indexed")) {
      /* the instanced rung's quad and shader: instance i at 2 + 16 i, green 32 (i + 1) */
      const bool indexed = strlen(name) > 10;
      const uint32_t first = indexed ? 1 : 2, count = indexed ? 3 : 2;
      s->shader = SH_INSTANCE;
      color_input(s);
      const float c[4] = { 120 / 255.0f, 0, 70 / 255.0f, 1 };
      const int vo = indexed ? 4 : 0;
      for (int i = 0; i < vo; i++)
         put(s, 0, (const float[6]){ 0 }, 6);
      quad_vertices(s, 2, 10, 14, 54, c);
      for (unsigned i = first; i < first + count; i++) {
         const float ci[4] = { c[0], (float)(32 * (i + 1)) / 255.0f, c[2], c[3] };
         fill(s, 2 + 16 * i, 10, 14 + 16 * i, 54, ci);
      }
      if (indexed) {
         s->index_size = 2;
         const uint32_t idx[6] = { 0, 1, 2, 2, 1, 3 };
         memcpy(s->indices, idx, sizeof(idx));
         s->index_count = 6;
         add_draw(s, 6, count, 0, vo);
      } else {
         s->topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
         add_draw(s, 4, count, 0, 0);
      }
      s->draw[0].first_instance = first;
   } else if (!strcmp(name, "inst-rate") || !strcmp(name, "inst-divisor0")) {
      /* binding 0: the quad (x 2..14, y 10..54), per vertex; binding 1: per instance (divisor 1) the offset
       * (inst-rate: and the colour); inst-divisor0: binding 2 the colour, divisor 0. Element k of binding 1
       * moves its instance to column k - 1; element 0 is never drawn */
      const bool d0 = name[5] == 'd';
      if (d0 && !has_divisor0) {
         *why = "no zero vertex attribute divisor";
         return false;
      }
      const uint32_t first = d0 ? 2 : 1, count = d0 ? 3 : 4;
      s->shader = SH_INST_ATTR;
      s->bindings = d0 ? 3 : 2;
      s->vib[0] = (VkVertexInputBindingDescription){ 0, 8, VK_VERTEX_INPUT_RATE_VERTEX };
      s->vib[1] = (VkVertexInputBindingDescription){ 1, d0 ? 8 : 24, VK_VERTEX_INPUT_RATE_INSTANCE };
      s->attribs = 3;
      s->via[0] = (VkVertexInputAttributeDescription){ 0, 0, VK_FORMAT_R32G32_SFLOAT, 0 };
      if (d0) {
         s->via[1] = (VkVertexInputAttributeDescription){ 1, 2, VK_FORMAT_R32G32B32A32_SFLOAT, 0 };
         s->via[2] = (VkVertexInputAttributeDescription){ 2, 1, VK_FORMAT_R32G32_SFLOAT, 0 };
         s->vib[2] = (VkVertexInputBindingDescription){ 2, 16, VK_VERTEX_INPUT_RATE_INSTANCE };
         s->divisor[s->ndivisors++] = (VkVertexInputBindingDivisorDescriptionKHR){ 2, 0 };
      } else {
         s->via[1] = (VkVertexInputAttributeDescription){ 1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0 };
         s->via[2] = (VkVertexInputAttributeDescription){ 2, 1, VK_FORMAT_R32G32_SFLOAT, 16 };
      }
      const int px[4] = { 2, 14, 2, 14 }, py[4] = { 10, 10, 54, 54 };
      for (unsigned k = 0; k < 4; k++)
         put(s, 0, (const float[2]){ nx(px[k]), ny(py[k]) }, 2);
      for (unsigned k = 0; k <= first + count; k++) {
         const float c[4] = { (float)(20 + 40 * k) / 255.0f, (float)(230 - 30 * k) / 255.0f, (float)(60 + 25 * k) / 255.0f,
                              1 };
         const float off[2] = { k ? (float)(16 * (k - 1)) / (W / 2) : 0.75f, 0 };
         if (d0) {
            put(s, 1, off, 2);
            put(s, 2, c, 4);
         } else {
            put(s, 1, c, 4);
            put(s, 1, off, 2);
         }
      }
      for (unsigned k = first; k < first + count; k++) {
         /* divisor 0: every instance takes element `first`'s colour */
         const unsigned ck = d0 ? first : k;
         const float c[4] = { (float)(20 + 40 * ck) / 255.0f, (float)(230 - 30 * ck) / 255.0f,
                              (float)(60 + 25 * ck) / 255.0f, 1 };
         fill(s, 2 + 16 * (k - 1), 10, 14 + 16 * (k - 1), 54, c);
      }
      s->topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
      add_draw(s, 4, count, 0, 0);
      s->draw[0].first_instance = first;
   } else if (!strncmp(name, "restart-", 8)) {
      /* quad q's 4 vertices (a strip of 2 triangles) at 4q, the strips joined by the cut index; without the cut
       * the strip would bridge the quads */
      const unsigned size = name[8] == '1' ? 2 : name[8] == '3' ? 4 : 1;
      if (size == 1 && !has_uint8) {
         *why = "no VK_KHR_index_type_uint8";
         return false;
      }
      const uint32_t cut = size == 1 ? 0xff : size == 2 ? 0xffff : 0xffffffffu;
      color_input(s);
      s->topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
      s->restart = true;
      s->index_size = size;
      for (unsigned q = 0; q < 8; q++) {
         int x0, y0, x1, y1;
         float c[4];
         grid_rect(q, &x0, &y0, &x1, &y1);
         quad_color(q, c);
         quad_vertices(s, x0, y0, x1, y1, c);
         fill(s, x0, y0, x1, y1, c);
         if (q)
            s->indices[s->index_count++] = cut;
         for (unsigned k = 0; k < 4; k++)
            s->indices[s->index_count++] = 4 * q + k;
      }
      add_draw(s, s->index_count, 1, 0, 0);
   } else if (!strncmp(name, "fmt-", 4)) {
      if (!fmt_build(name, s)) {
         *why = "no such rung (--list)";
         return false;
      }
      for (unsigned i = 0; i < ARRAY_LEN(s->need); i++) {
         if (!s->need[i])
            continue;
         VkFormatProperties fp;
         vkGetPhysicalDeviceFormatProperties(pd, s->need[i], &fp);
         if (!(fp.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT)) {
            static char buf[64];
            snprintf(buf, sizeof(buf), "format %d has no VERTEX_BUFFER feature", s->need[i]);
            *why = buf;
            return false;
         }
      }
   } else if (!strncmp(name, "int-", 4)) {
      /* one column of 8 quads (quad q at rows 8q+1..8q+7, columns 1..7) in binding 0, 16-bit indices;
       * 8 instances, instance i 8 pixels right: cell (i, q) has the shader's channels for q and i */
      static const char *const ops[] = { "imin", "imax", "umin", "umax", "isub", "ineg" };
      int op = -1;
      for (unsigned k = 0; k < ARRAY_LEN(ops); k++)
         if (!strcmp(name + 4, ops[k]))
            op = k;
      if (op < 0) {
         *why = "no such rung (--list)";
         return false;
      }
      s->shader = SH_INT_IMIN + op;
      s->push = true;
      memcpy(s->pc, INT_PC, sizeof(INT_PC));
      s->bindings = 1;
      s->vib[0] = (VkVertexInputBindingDescription){ 0, 8, VK_VERTEX_INPUT_RATE_VERTEX };
      s->attribs = 1;
      s->via[0] = (VkVertexInputAttributeDescription){ 0, 0, VK_FORMAT_R32G32_SFLOAT, 0 };
      s->index_size = 2;
      for (int q = 0; q < 8; q++) {
         const int y0 = 8 * q + 1, y1 = y0 + 6;
         const float v[4][2] = { { nx(1), ny(y0) }, { nx(7), ny(y0) }, { nx(1), ny(y1) }, { nx(7), ny(y1) } };
         for (unsigned k = 0; k < 4; k++)
            put(s, 0, v[k], 2);
         const uint32_t b = 4 * q;
         const uint32_t idx[6] = { b, b + 1, b + 2, b + 2, b + 1, b + 3 };
         memcpy(&s->indices[s->index_count], idx, sizeof(idx));
         s->index_count += 6;
         for (int i = 0; i < 8; i++) {
            const int32_t sq = q + INT_PC[0], ti = i + INT_PC[0];
            const uint32_t u = q, v = i, k5 = INT_PC[2];
            int64_t c[4];
            switch (op) {
            case 0: c[0] = MIN2(sq, ti) + 8; c[1] = MIN2(sq, 1) + 8; c[2] = MIN2(sq, INT_PC[1]) + 8;
                    c[3] = MIN2(ti, -2) + 8; break;
            case 1: c[0] = MAX2(sq, ti) + 8; c[1] = MAX2(sq, 1) + 8; c[2] = MAX2(sq, INT_PC[1]) + 8;
                    c[3] = MAX2(ti, -2) + 8; break;
            case 2: c[0] = MIN2(u, v); c[1] = MIN2(u, 5u); c[2] = MIN2(u, k5); c[3] = MIN2(u << 17, 0x50000u) >> 12;
                    break;
            case 3: c[0] = MAX2(u, v); c[1] = MAX2(u, 3u); c[2] = MAX2(u, k5); c[3] = MAX2(u << 17, 0x50000u) >> 12;
                    break;
            case 4: c[0] = sq - ti + 8; c[1] = 100 - q; c[2] = INT_PC[3] - q + 8; c[3] = q - INT_PC[3] + 8; break;
            default: c[0] = -sq + 8; c[1] = (sq < 0 ? -sq : sq) + 8; c[2] = -ti + 8; c[3] = -q + 8; break;
            }
            const float rgba[4] = { c[0] / 255.0f, c[1] / 255.0f, c[2] / 255.0f, c[3] / 255.0f };
            fill(s, 8 * i + 1, y0, 8 * i + 7, y1, rgba);
         }
      }
      add_draw(s, 48, 8, 0, 0);
   } else if (!strcmp(name, "strip")) {
      /* one strip of 3 quads across the target, one colour */
      s->topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
      color_input(s);
      const float c[4] = { 10 / 255.0f, 220 / 255.0f, 130 / 255.0f, 1 };
      const int xs[4] = { 4, 20, 40, 60 };
      for (unsigned i = 0; i < 4; i++) {
         put(s, 0, (const float[6]){ nx(xs[i]), ny(8), c[0], c[1], c[2], c[3] }, 6);
         put(s, 0, (const float[6]){ nx(xs[i]), ny(56), c[0], c[1], c[2], c[3] }, 6);
      }
      fill(s, 4, 8, 60, 56, c);
      add_draw(s, 8, 1, 0, 0);
   } else if (!strcmp(name, "fan")) {
      /* two fans of 4 vertices (a quad each, drawn as 2 triangles around corner 0) */
      s->topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
      color_input(s);
      for (unsigned q = 0; q < 2; q++) {
         float c[4];
         const int x0 = 4 + 30 * q, x1 = x0 + 26, y0 = 6, y1 = 58;
         quad_color(q + 2, c);
         const int px[4] = { x0, x1, x1, x0 }, py[4] = { y0, y0, y1, y1 };
         for (unsigned i = 0; i < 4; i++)
            put(s, 0, (const float[6]){ nx(px[i]), ny(py[i]), c[0], c[1], c[2], c[3] }, 6);
         fill(s, x0, y0, x1, y1, c);
         add_draw(s, 4, 1, 4 * q, 0);
      }
   } else if (!strcmp(name, "lines") || !strcmp(name, "line-strip")) {
      const bool strip = name[4] == '-';
      s->topology = strip ? VK_PRIMITIVE_TOPOLOGY_LINE_STRIP : VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
      color_input(s);
      const float c[4] = { 250 / 255.0f, 20 / 255.0f, 90 / 255.0f, 1 };
      if (strip) {
         /* a zigzag of horizontal runs joined by vertical steps: only the horizontal runs are compared */
         const int ys[4] = { 10, 25, 40, 55 };
         for (unsigned i = 0; i < 4; i++) {
            const float y = ny(ys[i]) + 1.0f / H; /* the row's centre */
            put(s, 0, (const float[6]){ nx(i % 2 ? 58 : 6), y, c[0], c[1], c[2], c[3] }, 6);
            put(s, 0, (const float[6]){ nx(i % 2 ? 6 : 58), y, c[0], c[1], c[2], c[3] }, 6);
            hline(s, 6, 58, ys[i], c);
         }
         /* the vertical joins (x = 6 or 58, between the rows): not compared */
         for (int y = 9; y <= 56; y++)
            for (int x = 4; x <= 7; x++)
               s->skip[y][x] = s->skip[y][x + 52] = true;
         add_draw(s, 8, 1, 0, 0);
      } else {
         for (unsigned i = 0; i < 6; i++) {
            const int y = 6 + 10 * i;
            const float yc = ny(y) + 1.0f / H;
            put(s, 0, (const float[6]){ nx(5), yc, c[0], c[1], c[2], c[3] }, 6);
            put(s, 0, (const float[6]){ nx(59), yc, c[0], c[1], c[2], c[3] }, 6);
            hline(s, 5, 59, y, c);
         }
         add_draw(s, 12, 1, 0, 0);
      }
   } else {
      *why = "no such rung (--list)";
      return false;
   }
   return true;
}

static int
host_memory(const VkMemoryRequirements *req, VkDeviceMemory *mem)
{
   VkPhysicalDeviceMemoryProperties mp;
   uint32_t type = UINT32_MAX;

   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount && type == UINT32_MAX; i++) {
      if ((req->memoryTypeBits & (1u << i)) &&
          (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
         type = i;
   }
   if (type == UINT32_MAX) {
      printf("VERTEX-INPUT %s FAIL: no host-visible memory type\n", cur);
      return 1;
   }
   const VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                      .allocationSize = req->size, .memoryTypeIndex = type };
   CHECK(vkAllocateMemory(dev, &mai, NULL, mem));
   return 0;
}

static int
flush(VkDeviceMemory mem, bool invalidate)
{
   const VkMappedMemoryRange range = { .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, .memory = mem,
                                       .size = VK_WHOLE_SIZE };

   if (invalidate)
      CHECK(vkInvalidateMappedMemoryRanges(dev, 1, &range));
   else
      CHECK(vkFlushMappedMemoryRanges(dev, 1, &range));
   return 0;
}

/* A host-visible buffer holding `bytes` of data */
struct buf {
   VkBuffer buffer;
   VkDeviceMemory mem;
};

static int
make_buffer(struct buf *b, VkBufferUsageFlags usage, const void *data, size_t bytes)
{
   const VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes ? bytes : 4,
                                    .usage = usage };
   VkMemoryRequirements req;
   void *map;

   CHECK(vkCreateBuffer(dev, &bci, NULL, &b->buffer));
   vkGetBufferMemoryRequirements(dev, b->buffer, &req);
   if (host_memory(&req, &b->mem))
      return 1;
   CHECK(vkBindBufferMemory(dev, b->buffer, b->mem, 0));
   CHECK(vkMapMemory(dev, b->mem, 0, VK_WHOLE_SIZE, 0, &map));
   memcpy(map, data, bytes);
   vkUnmapMemory(dev, b->mem);
   return flush(b->mem, false);
}

static void
free_buffer(struct buf *b)
{
   vkDestroyBuffer(dev, b->buffer, NULL);
   vkFreeMemory(dev, b->mem, NULL);
}

static struct scene scene;

/* 0 pass, 1 fail, 2 skip */
static int
run(const struct rung *r)
{
   struct scene *s = &scene;
   const char *why = NULL;

   cur = r->name;
   if (!build(r->name, s, &why)) {
      printf("VERTEX-INPUT %s SKIP: %s\n", r->name, why);
      return 2;
   }

   /* the target */
   const VkImageCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                                   .format = VK_FORMAT_B8G8R8A8_UNORM, .extent = { W, H, 1 }, .mipLevels = 1,
                                   .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                                   .tiling = VK_IMAGE_TILING_LINEAR, .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                                   .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
   VkImage image;
   VkDeviceMemory imem;
   VkMemoryRequirements ireq;
   VkSubresourceLayout il;
   uint8_t *imap;
   CHECK(vkCreateImage(dev, &ici, NULL, &image));
   vkGetImageMemoryRequirements(dev, image, &ireq);
   if (host_memory(&ireq, &imem))
      return 1;
   CHECK(vkBindImageMemory(dev, image, imem, 0));
   vkGetImageSubresourceLayout(dev, image, &(VkImageSubresource){ VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 }, &il);
   CHECK(vkMapMemory(dev, imem, 0, VK_WHOLE_SIZE, 0, (void **)&imap));
   memset(imap, 0x5a, ireq.size);
   if (flush(imem, false))
      return 1;
   const VkImageViewCreateInfo vci = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = image,
                                       .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_B8G8R8A8_UNORM,
                                       .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
   VkImageView view;
   CHECK(vkCreateImageView(dev, &vci, NULL, &view));

   /* render pass: clear, store; the layout ends GENERAL for the host read */
   const VkAttachmentDescription att = { .format = VK_FORMAT_B8G8R8A8_UNORM, .samples = VK_SAMPLE_COUNT_1_BIT,
                                         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                                         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                         .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                         .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                         .finalLayout = VK_IMAGE_LAYOUT_GENERAL };
   const VkAttachmentReference ref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   const VkSubpassDescription sp = { .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     .colorAttachmentCount = 1, .pColorAttachments = &ref };
   const VkRenderPassCreateInfo rci = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, .attachmentCount = 1,
                                        .pAttachments = &att, .subpassCount = 1, .pSubpasses = &sp };
   VkRenderPass rp;
   CHECK(vkCreateRenderPass(dev, &rci, NULL, &rp));
   const VkFramebufferCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = rp,
                                         .attachmentCount = 1, .pAttachments = &view, .width = W, .height = H,
                                         .layers = 1 };
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(dev, &fci, NULL, &fb));

   /* vertex and index buffers */
   struct buf vb[MAX_BINDINGS], ib = { 0 };
   for (unsigned b = 0; b < s->bindings; b++)
      if (make_buffer(&vb[b], VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, s->data[b], s->data_floats[b] * sizeof(float)))
         return 1;
   if (s->index_size) {
      uint8_t bytes[64 * 4];
      for (unsigned i = 0; i < s->index_count; i++) {
         if (s->index_size == 1)
            bytes[i] = (uint8_t)s->indices[i];
         else if (s->index_size == 2)
            ((uint16_t *)bytes)[i] = (uint16_t)s->indices[i];
         else
            ((uint32_t *)bytes)[i] = s->indices[i];
      }
      if (make_buffer(&ib, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, bytes, s->index_count * s->index_size))
         return 1;
   }

   /* pipeline */
   const uint32_t *vs_code;
   size_t vs_size;
   switch (s->shader) {
   case SH_INSTANCE: vs_code = spv_instance_vert; vs_size = sizeof(spv_instance_vert); break;
   case SH_VERTEX_INDEX: vs_code = spv_vertex_index_vert; vs_size = sizeof(spv_vertex_index_vert); break;
   case SH_ATTRIB16: vs_code = spv_attrib16_vert; vs_size = sizeof(spv_attrib16_vert); break;
   case SH_INT_IMIN: vs_code = spv_int_imin_vert; vs_size = sizeof(spv_int_imin_vert); break;
   case SH_INT_IMAX: vs_code = spv_int_imax_vert; vs_size = sizeof(spv_int_imax_vert); break;
   case SH_INT_UMIN: vs_code = spv_int_umin_vert; vs_size = sizeof(spv_int_umin_vert); break;
   case SH_INT_UMAX: vs_code = spv_int_umax_vert; vs_size = sizeof(spv_int_umax_vert); break;
   case SH_INT_ISUB: vs_code = spv_int_isub_vert; vs_size = sizeof(spv_int_isub_vert); break;
   case SH_INT_INEG: vs_code = spv_int_ineg_vert; vs_size = sizeof(spv_int_ineg_vert); break;
   case SH_FMT: vs_code = spv_fmt_vert; vs_size = sizeof(spv_fmt_vert); break;
   case SH_FMT_INT: vs_code = spv_fmt_int_vert; vs_size = sizeof(spv_fmt_int_vert); break;
   case SH_INST_ATTR: vs_code = spv_inst_attr_vert; vs_size = sizeof(spv_inst_attr_vert); break;
   default: vs_code = spv_color_vert; vs_size = sizeof(spv_color_vert); break;
   }
   VkShaderModuleCreateInfo smi = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = vs_size,
                                    .pCode = vs_code };
   VkShaderModule vsm, fsm;
   CHECK(vkCreateShaderModule(dev, &smi, NULL, &vsm));
   smi.codeSize = sizeof(spv_color_frag);
   smi.pCode = spv_color_frag;
   CHECK(vkCreateShaderModule(dev, &smi, NULL, &fsm));
   const VkPushConstantRange pcr = { VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(s->pc) };
   const VkPipelineLayoutCreateInfo lci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                            .pushConstantRangeCount = s->push, .pPushConstantRanges = &pcr };
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(dev, &lci, NULL, &layout));
   const VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vsm, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = fsm, .pName = "main" },
   };
   const VkPipelineVertexInputDivisorStateCreateInfoKHR vdiv = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_KHR,
      .vertexBindingDivisorCount = s->ndivisors, .pVertexBindingDivisors = s->divisor };
   const VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, .pNext = s->ndivisors ? &vdiv : NULL,
      .vertexBindingDescriptionCount = s->bindings, .pVertexBindingDescriptions = s->vib,
      .vertexAttributeDescriptionCount = s->attribs, .pVertexAttributeDescriptions = s->via };
   const VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = s->topology,
      .primitiveRestartEnable = s->restart };
   const VkPipelineViewportStateCreateInfo vps = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                                   .viewportCount = 1, .scissorCount = 1 };
   const VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE, .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f };
   const VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
   const VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xf };
   const VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1,
      .pAttachments = &cba };
   const VkDynamicState dyns[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
   const VkPipelineDynamicStateCreateInfo ds = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                                 .dynamicStateCount = 2, .pDynamicStates = dyns };
   const VkGraphicsPipelineCreateInfo gci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = 2, .pStages = stages,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vps,
      .pRasterizationState = &rs, .pMultisampleState = &ms, .pColorBlendState = &cb, .pDynamicState = &ds,
      .layout = layout, .renderPass = rp };
   VkPipeline pipe;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gci, NULL, &pipe));

   /* record and submit */
   const VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                         .queueFamilyIndex = queue_family };
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(dev, &pci, NULL, &pool));
   const VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                             .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                             .commandBufferCount = 1 };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
   const VkCommandBufferBeginInfo cbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
   CHECK(vkBeginCommandBuffer(cmd, &cbi));
   const VkClearValue cv = { .color.float32 = { CLEAR[0], CLEAR[1], CLEAR[2], CLEAR[3] } };
   const VkRenderPassBeginInfo rbi = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = rp,
                                       .framebuffer = fb, .renderArea = { { 0, 0 }, { W, H } },
                                       .clearValueCount = 1, .pClearValues = &cv };
   vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   VkBuffer vbs[MAX_BINDINGS];
   for (unsigned b = 0; b < s->bindings; b++)
      vbs[b] = vb[b].buffer;
   vkCmdBindVertexBuffers(cmd, 0, s->bindings, vbs, s->bind_offset);
   if (s->push)
      vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(s->pc), s->pc);
   if (s->index_size)
      vkCmdBindIndexBuffer(cmd, ib.buffer, 0,
                           s->index_size == 1   ? VK_INDEX_TYPE_UINT8_EXT
                           : s->index_size == 2 ? VK_INDEX_TYPE_UINT16
                                                : VK_INDEX_TYPE_UINT32);
   const VkViewport vpt = { 0, 0, W, H, 0, 1 };
   const VkRect2D sc = { { 0, 0 }, { W, H } };
   vkCmdSetViewport(cmd, 0, 1, &vpt);
   vkCmdSetScissor(cmd, 0, 1, &sc);
   for (unsigned d = 0; d < s->draws; d++) {
      if (s->index_size)
         vkCmdDrawIndexed(cmd, s->draw[d].count, s->draw[d].instances, s->draw[d].first, s->draw[d].vertex_offset,
                          s->draw[d].first_instance);
      else
         vkCmdDraw(cmd, s->draw[d].count, s->draw[d].instances, s->draw[d].first, s->draw[d].first_instance);
   }
   vkCmdEndRenderPass(cmd);
   CHECK(vkEndCommandBuffer(cmd));
   const VkFenceCreateInfo fnci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
   VkFence fence;
   CHECK(vkCreateFence(dev, &fnci, NULL, &fence));
   const VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                             .pCommandBuffers = &cmd };
   CHECK(vkQueueSubmit(queue, 1, &si, fence));
   CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 5000000000ull));
   if (flush(imem, true))
      return 1;

   /* compare: the target stores B, G, R, A */
   unsigned bad = 0, fx = 0, fy = 0, compared = 0;
   uint8_t got_f[4] = { 0 }, want_f[4] = { 0 };
   for (unsigned y = 0; verify && y < H; y++) {
      for (unsigned x = 0; x < W; x++) {
         if (s->skip[y][x])
            continue;
         const uint8_t *p = imap + il.offset + y * il.rowPitch + 4 * x;
         const uint8_t got[4] = { p[2], p[1], p[0], p[3] };
         compared++;
         if (memcmp(got, s->want[y][x], 4) && !bad++) {
            fx = x, fy = y;
            memcpy(got_f, got, 4);
            memcpy(want_f, s->want[y][x], 4);
         }
      }
   }
   if (!verify)
      printf("VERTEX-INPUT %s PASS: submitted and retired (not compared)\n", r->name);
   else if (bad)
      printf("VERTEX-INPUT %s FAIL: %u of %u pixels differ, first at %u,%u: RGBA %u %u %u %u, expected %u %u %u %u\n",
             r->name, bad, compared, fx, fy, got_f[0], got_f[1], got_f[2], got_f[3], want_f[0], want_f[1],
             want_f[2], want_f[3]);
   else
      printf("VERTEX-INPUT %s PASS: %u pixels exact (%u not compared)\n", r->name, compared, W * H - compared);

   vkDestroyFence(dev, fence, NULL);
   vkFreeCommandBuffers(dev, pool, 1, &cmd);
   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   vkDestroyShaderModule(dev, vsm, NULL);
   vkDestroyShaderModule(dev, fsm, NULL);
   for (unsigned b = 0; b < s->bindings; b++)
      free_buffer(&vb[b]);
   if (s->index_size)
      free_buffer(&ib);
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyRenderPass(dev, rp, NULL);
   vkDestroyImageView(dev, view, NULL);
   vkUnmapMemory(dev, imem);
   vkDestroyImage(dev, image, NULL);
   vkFreeMemory(dev, imem, NULL);
   return verify && bad ? 1 : 0;
}

int
main(int argc, char **argv)
{
   unsigned device_index = 0, nsel = 0, pass = 0, ran = 0;
   const char *sel[ARRAY_LEN(rungs) + 8];

   setvbuf(stdout, NULL, _IOLBF, 0);
   for (int a = 1; a < argc; a++) {
      if (!strcmp(argv[a], "--list")) {
         for (unsigned i = 0; i < ARRAY_LEN(rungs); i++)
            printf("%s\n", rungs[i].name);
         return 0;
      } else if (!strcmp(argv[a], "--device") && a + 1 < argc) {
         device_index = (unsigned)atoi(argv[++a]);
      } else if (!strcmp(argv[a], "--no-verify")) {
         verify = false;
      } else if (argv[a][0] != '-' && nsel < ARRAY_LEN(sel)) {
         sel[nsel++] = argv[a];
      } else {
         fprintf(stderr, "usage: %s [--list] [--device N] [--no-verify] [RUNG...]\n", argv[0]);
         return 2;
      }
   }

   const VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                   .pApplicationName = "vertex-input", .apiVersion = VK_API_VERSION_1_1 };
   const VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                      .pApplicationInfo = &app };
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 0;
   CHECK(vkEnumeratePhysicalDevices(inst, &n, NULL));
   VkPhysicalDevice pds[16];
   n = n > 16 ? 16 : n;
   CHECK(vkEnumeratePhysicalDevices(inst, &n, pds));
   if (device_index >= n) {
      printf("VERTEX-INPUT RESULT FAIL: no Vulkan device %u (%u found)\n", device_index, n);
      return 1;
   }
   pd = pds[device_index];
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("VERTEX-INPUT device %u: %s, Vulkan %u.%u, driver %#x\n", device_index, props.deviceName,
          VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion), props.driverVersion);

   /* 8-bit indices: VK_KHR_index_type_uint8 (or EXT) and its feature */
   const char *ext = NULL, *dext = NULL;
   uint32_t ne = 0;
   vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, NULL);
   VkExtensionProperties *exts = calloc(ne ? ne : 1, sizeof(*exts));
   vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, exts);
   for (uint32_t i = 0; i < ne && !ext; i++) {
      if (!strcmp(exts[i].extensionName, "VK_KHR_index_type_uint8"))
         ext = "VK_KHR_index_type_uint8";
   }
   for (uint32_t i = 0; i < ne && !ext; i++) {
      if (!strcmp(exts[i].extensionName, "VK_EXT_index_type_uint8"))
         ext = "VK_EXT_index_type_uint8";
   }
   for (uint32_t i = 0; i < ne; i++) {
      if (!strcmp(exts[i].extensionName, "VK_KHR_vertex_attribute_divisor") ||
          (!dext && !strcmp(exts[i].extensionName, "VK_EXT_vertex_attribute_divisor")))
         dext = exts[i].extensionName[3] == 'K' ? "VK_KHR_vertex_attribute_divisor" : "VK_EXT_vertex_attribute_divisor";
   }
   free(exts);
   /* the zero divisor (inst-divisor0) */
   VkPhysicalDeviceVertexAttributeDivisorFeaturesKHR divf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_KHR };
   if (dext) {
      VkPhysicalDeviceFeatures2 f2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &divf };
      vkGetPhysicalDeviceFeatures2(pd, &f2);
   }
   has_divisor0 = dext && divf.vertexAttributeInstanceRateZeroDivisor;
   divf.vertexAttributeInstanceRateDivisor = divf.vertexAttributeInstanceRateZeroDivisor = has_divisor0;
   VkPhysicalDeviceIndexTypeUint8FeaturesEXT u8 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT,
                                                    .indexTypeUint8 = VK_TRUE };
   has_uint8 = ext != NULL;

   VkQueueFamilyProperties qf[8];
   uint32_t nq = 8;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf);
   queue_family = UINT32_MAX;
   for (uint32_t i = 0; i < nq && queue_family == UINT32_MAX; i++) {
      if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
         queue_family = i;
   }
   if (queue_family == UINT32_MAX) {
      printf("VERTEX-INPUT RESULT FAIL: no graphics queue\n");
      return 1;
   }
   const float prio = 1.0f;
   const VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                         .queueFamilyIndex = queue_family, .queueCount = 1,
                                         .pQueuePriorities = &prio };
   const char *enable[2];
   uint32_t nen = 0;
   if (has_uint8)
      enable[nen++] = ext;
   if (has_divisor0)
      enable[nen++] = dext;
   divf.pNext = has_uint8 ? &u8 : NULL;
   const VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                    .pNext = has_divisor0 ? (void *)&divf : has_uint8 ? (void *)&u8 : NULL,
                                    .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
                                    .enabledExtensionCount = nen, .ppEnabledExtensionNames = enable };
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   vkGetDeviceQueue(dev, queue_family, 0, &queue);

   for (unsigned i = 0; i < ARRAY_LEN(rungs); i++) {
      bool want = !nsel;
      for (unsigned k = 0; k < nsel; k++)
         want |= !strcmp(sel[k], rungs[i].name);
      if (!want)
         continue;
      const int rc = run(&rungs[i]);
      if (rc != 2) {
         ran++;
         pass += rc == 0;
      }
   }
   for (unsigned k = 0; k < nsel; k++) {
      bool known = false;
      for (unsigned i = 0; i < ARRAY_LEN(rungs); i++)
         known |= !strcmp(sel[k], rungs[i].name);
      if (!known) {
         printf("VERTEX-INPUT %s FAIL: no such rung (--list)\n", sel[k]);
         ran++;
      }
   }
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   printf("VERTEX-INPUT RESULT %s %u of %u\n", ran && pass == ran ? "PASS" : "FAIL", pass, ran);
   return ran && pass == ran ? 0 : 1;
}
