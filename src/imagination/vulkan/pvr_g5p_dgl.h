/* SPDX-License-Identifier: MIT
 * Generated-by: LLM
 * Integer fragment-output building block. Does not advertise format support.
 */
#ifndef PVR_G5P_DGL_H
#define PVR_G5P_DGL_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct pvr_g5p_dgl_source {
   bool immediate;
   uint32_t value; /* literal bits, or allocated temporary register index */
   unsigned group; /* register last-read group, as selected by the backend */
};

/* Four unconverted 32-bit channels into o0..o3, emitpix(4), end.
 * The caller must preserve source temporary lifetimes until this epilogue.
 * Returns zero for an unencodable source or insufficient output capacity.
 * Writes nothing to output on failure. This is not the packed-f16 path.
 */
size_t pvr_g5p_dgl_output(uint8_t *output, size_t capacity,
                         const struct pvr_g5p_dgl_source sources[4], unsigned write_mask);
/* Background clear: raw common-store dwords, no float conversion. */
size_t pvr_g5p_dgl_clear(uint8_t *output, size_t capacity);
size_t pvr_g5p_dgl_clear_words(uint8_t *output, size_t capacity, unsigned words);
void pvr_g5p_dgl_clear_pack(uint32_t output[4], const uint32_t input[4], bool uint16);
#endif
