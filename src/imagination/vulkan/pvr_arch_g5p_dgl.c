/* SPDX-License-Identifier: MIT
 * Generated-by: LLM
 * A9 integer colour output: no floating-point conversion of source bits.
 * Same physical output register class as PCO PIXOUT, and the
 * existing pco_g5p.c constant FS with half_output=false (o0..o3, emitpix 4).
 * Register encodings are the existing G5P field encoders, not program arrays.
 * This component deliberately does not change format/feature advertisement.
 */
#include "pvr_g5p_dgl.h"
#include "g5p_usc.h"
#include <string.h>

/* words: 4 for one 32-bit integer target, 8 for two (target t's channels in
 * output words 4t..4t+3, one emitpix of all eight: the width of Apple's
 * 4-target pass). Two targets take the full write mask only. */
size_t
pvr_g5p_dgl_output_words(uint8_t *output, size_t capacity, const struct pvr_g5p_dgl_source *sources,
                         unsigned words, unsigned write_mask)
{
   uint8_t code[10 * G5P_USC_MAX];
   size_t size = 0;
   if (!output || !sources || write_mask > 0xf || (words != 4 && words != 8) || (words == 8 && write_mask != 0xf))
      return 0;
   if (write_mask != 0xf) {
      /* RGBA32 tile storage is four words. Preserve unmasked raw channels;
       * unlike the F16 blend path, no half conversion is requested here.
       * Hardware qualification is required before exposing these formats. */
      size += g5p_usc_tile_read(code + size, 0, 4);
      size += g5p_usc_wait(code + size, 0);
   }
   for (unsigned c = 0; c < words; c++) {
      if (!(write_mask & (1u << (c % 4))))
         continue;
      const struct pvr_g5p_dgl_source *s = &sources[c];
      unsigned n;
      if (s->immediate) {
         n = g5p_usc_immediate(code + size, 2 * c, s->value, 0, -1);
      } else {
         /* This selector addresses the backend's low temporary bank. A high
          * home must be materialized into a low scratch before calling us. */
         if (s->value > 31 || s->group > 1)
            return 0;
         n = g5p_usc_move(code + size, 2 * c, 2 * s->value, 0, s->group, 1);
      }
      if (!n)
         return 0;
      size += n;
   }
   unsigned n = g5p_usc_emit_mrt(code + size, words, 0);
   if (!n)
      return 0;
   size += n;
   size += g5p_usc_special(code + size, true);
   if (capacity < size)
      return 0;
   memcpy(output, code, size);
   return size;
}

size_t
pvr_g5p_dgl_output(uint8_t *output, size_t capacity,
                    const struct pvr_g5p_dgl_source sources[4], unsigned write_mask)
{
   return pvr_g5p_dgl_output_words(output, capacity, sources, 4, write_mask);
}

/* Background objects have a distinct ABI from ordinary fragment uniforms:
 * their DOUTD supplies common0 onward. Apple A9 builtin compiler witnesses
 * use descending common3..0 for RGBA32 and common1..0 for RG32.
 */
size_t
pvr_g5p_dgl_clear_words(uint8_t *output, size_t capacity, unsigned words)
{
   uint8_t code[10 * G5P_USC_MAX];
   size_t size = 0;
   /* 8: two 32-bit integer targets, common 7..0 into output words 7..0 */
   if (words != 2 && words != 4 && words != 8)
      return 0;
   for (int c = words - 1; c >= 0; c--)
      size += g5p_usc_mov_shared(code + size, 2 * c, c, 0);
   size += g5p_usc_emit_mrt(code + size, words, 0);
   size += g5p_usc_special(code + size, true);
   if (!output || capacity < size)
      return 0;
   memcpy(output, code, size);
   return size;
}

size_t
pvr_g5p_dgl_clear(uint8_t *output, size_t capacity)
{
   return pvr_g5p_dgl_clear_words(output, capacity, 4);
}

/* Vulkan integer clear conversion can be done before the background DMA.
 * RGBA16UI clamps each u32 channel to UINT16_MAX, then packs two per word;
 * it never passes through a floating-point or half-float conversion.
 */
void
pvr_g5p_dgl_clear_pack(uint32_t output[4], const uint32_t input[4], bool uint16)
{
   uint32_t value[4];
   for (unsigned c = 0; c < 4; c++)
      value[c] = uint16 && input[c] > 65535 ? 65535 : input[c];
   if (uint16) {
      output[0] = value[0] | value[1] << 16;
      output[1] = value[2] | value[3] << 16;
      output[2] = output[3] = 0;
   } else {
      memcpy(output, value, sizeof(value));
   }
}
