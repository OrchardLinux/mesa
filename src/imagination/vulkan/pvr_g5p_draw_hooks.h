/* SPDX-License-Identifier: MIT */
/*
 * The Apple A9 (G5P) draw hooks in pvr_arch_cmd_buffer.c, in the form of
 * pvr_g5p_hooks.h (one PVR_G5P_HOOK(name) line that expands to nothing
 * without -Dimagination-g5p-experimental).
 *
 * The indirect lane wrote these in pvr_g5p_publish_hooks.h, which the
 * desktop perf base (ram3, insertion-only hooks) no longer has; they are
 * carried here unchanged (lane tip f610adc2213). Both expand inside
 * pvr_CmdDrawIndexedIndirect()/pvr_CmdDrawIndirect() and use that
 * function's cmd_buffer, state, buffer, offset, drawCount, stride and
 * result.
 */

#ifndef PVR_G5P_DRAW_HOOKS_H
#define PVR_G5P_DRAW_HOOKS_H

#if defined(PVR_APPLE_G5P)
#include "pvr_g5p.h"

/* Route indirect commands through the G5P builder, never generic Rogue PDS. */
#define PVR_G5P_HOOK_g5p_indexed_indirect() \
   if (PVR_HAS_FEATURE(&cmd_buffer->device->pdevice->dev_info, apple_g5p)) { \
      result = pvr_flush_push_descriptors(cmd_buffer, &state->gfx_desc_state); \
      if (result == VK_SUCCESS) \
         result = pvr_arch_g5p_cmd_draw_indirect(cmd_buffer, buffer, offset, drawCount, stride, true); \
      if (result != VK_SUCCESS) \
         vk_command_buffer_set_error(&cmd_buffer->vk, result); \
      goto end_cmd_draw_indexed_indirect; \
   }
#define PVR_G5P_HOOK_g5p_indirect() \
   if (PVR_HAS_FEATURE(&cmd_buffer->device->pdevice->dev_info, apple_g5p)) { \
      result = pvr_flush_push_descriptors(cmd_buffer, &state->gfx_desc_state); \
      if (result == VK_SUCCESS) \
         result = pvr_arch_g5p_cmd_draw_indirect(cmd_buffer, buffer, offset, drawCount, stride, false); \
      if (result != VK_SUCCESS) \
         vk_command_buffer_set_error(&cmd_buffer->vk, result); \
      goto end_cmd_draw_indirect; \
   }
#endif

#endif /* PVR_G5P_DRAW_HOOKS_H */
