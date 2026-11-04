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

size_t
pvr_g5p_dgl_output(uint8_t *output, size_t capacity,
                    const struct pvr_g5p_dgl_source sources[4], unsigned write_mask)
{
   uint8_t code[6 * G5P_USC_MAX];
   size_t size = 0;
   if (!output || !sources || write_mask > 0xf)
      return 0;
   if (write_mask != 0xf) {
      /* RGBA32 tile storage is four words. Preserve unmasked raw channels;
       * unlike the F16 blend path, no half conversion is requested here.
       * Hardware qualification is required before exposing these formats. */
      size += g5p_usc_tile_read(code + size, 0, 4);
      size += g5p_usc_wait(code + size, 0);
   }
   for (unsigned c = 0; c < 4; c++) {
      if (!(write_mask & (1u << c)))
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
   unsigned n = g5p_usc_emit_mrt(code + size, 4, 0);
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
pvr_g5p_dgl_clear(uint8_t *output, size_t capacity)
{
   uint8_t code[6 * G5P_USC_MAX];
   size_t size = 0;
   /* Apple A9 raw uint/int output: descending output writes from common2..5.
    * The ordinary compiler also places shared-move values at dword2 or above.
    * The caller must DMA the four raw clear dwords at common2 and allocate six. */
   for (int c = 3; c >= 0; c--) {
      unsigned n = g5p_usc_mov_shared(code + size, 2 * c, c + 2, 0);
      if (!n)
         return 0;
      size += n;
   }
   size += g5p_usc_emit_mrt(code + size, 4, 0);
   size += g5p_usc_special(code + size, true);
   if (!output || capacity < size)
      return 0;
   memcpy(output, code, size);
   return size;
}
