/* SPDX-License-Identifier: MIT */
/*
 * A fast ETC2 encoder for 8-bit RGBA texels: ETC2 RGB8, RGB8 with punchthrough
 * alpha, and RGBA8 (EAC alpha), linear or sRGB.
 */

#ifndef U_FORMAT_ETC2_ENC_H
#define U_FORMAT_ETC2_ENC_H

#include <stdbool.h>
#include <stdint.h>

#include "util/format/u_formats.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Whether util_format_etc2_encode_rgba8() writes this format */
bool
util_format_etc2_can_encode(enum pipe_format format);

/*
 * Encode width x height texels of R8G8B8A8 (the sRGB formats take sRGB
 * values) into rows of 4x4 blocks of an ETC2 format. A block past the right
 * or bottom edge repeats the edge texels. With punchthrough alpha a texel is
 * transparent when its alpha is below 128.
 */
void
util_format_etc2_encode_rgba8(enum pipe_format format,
                              uint8_t *dst_row, unsigned dst_stride,
                              const uint8_t *src_row, unsigned src_stride,
                              unsigned width, unsigned height);

#ifdef __cplusplus
}
#endif

#endif /* U_FORMAT_ETC2_ENC_H */
