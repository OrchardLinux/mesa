/* SPDX-License-Identifier: MIT
 * Entry points of the G5P (Apple A9) backend's captured-shape matcher
 * (pco_g5p.c), formerly declared in pco/pco.h; the general backend is in
 * pco_g5p_general.h.
 */
#ifndef PCO_G5P_H
#define PCO_G5P_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "compiler/nir/nir.h"

bool pco_g5p_compile_simple(nir_shader *nir, uint8_t **data, size_t *size, bool half_output);
enum { PCO_G5P_SHAPE_VS_PASSTHROUGH = 1, PCO_G5P_SHAPE_VS_VARYING,
       PCO_G5P_SHAPE_FS_CONSTANT, PCO_G5P_SHAPE_FS_VARYING };
int pco_g5p_compile_shape(nir_shader *nir, uint8_t **data, size_t *size, bool half_output,
                          unsigned *varyings);

#endif /* PCO_G5P_H */
