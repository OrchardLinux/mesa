/* SPDX-License-Identifier: MIT */
/* PR1-style hooks: each call consumes one original blank line before the next
 * line-sensitive statement. Definitions capture the call site locals. No #line. */
#ifndef PVR_G9P_ES3_ARCH_G9P_GFX_HOOKS_H
#define PVR_G9P_ES3_ARCH_G9P_GFX_HOOKS_H
#include "pvr_g9p_es3_hooks.h"
#ifdef PVR_G9P_ES3

#include "g9p_es3_isa.h"

struct g9p_draw_programs;
/* -Dimagination-g9p-es3, at the end of this file (pvr_arch_g9p_es3_gfx.h) */
static VkResult es3_gfx_device_init(struct pvr_device *device, struct g9p_code *c);
static void es3_gfx_device_finish(struct pvr_device *device);
static void es3_trim_outputs(nir_shader *fs, const struct vk_graphics_pipeline_state *vk_state);
static VkResult es3_check_targets(struct pvr_device *device,
                                  const struct vk_graphics_pipeline_state *vk_state,
                                  const struct g9p_shader *fs);
static VkResult es3_job_init(struct pvr_cmd_buffer *cmd_buffer,
                             const struct pvr_renderpass_hwsetup_render *hw_render,
                             struct pvr_render_job *job, bool *handled);
static void es3_frag_stream_patch(const struct pvr_render_job *job, uint8_t *buf,
                                  uint32_t info_at, uint32_t cluster1_at, uint32_t r4070_at);
static VkResult es3_clear_attachments(struct pvr_cmd_buffer *cmd_buffer,
                                      uint32_t attachment_count,
                                      const VkClearAttachment *attachments, uint32_t rect_count,
                                      const VkClearRect *rects, bool *handled);

/* pvr_arch_g9p_gfx.c, after main line 288: the option-on insertion. */
#define PVR_G9P_ES3_HOOK_arch_g9p_gfx_288() \
   if (result == VK_SUCCESS) \
   result = es3_gfx_device_init(device, c);

/* pvr_arch_g9p_gfx.c, after main line 321: the option-on insertion. */
#define PVR_G9P_ES3_HOOK_arch_g9p_gfx_321() \
   es3_gfx_device_finish(device);

/* pvr_arch_g9p_gfx.c, after main line 520: the option-on insertion. */
#define PVR_G9P_ES3_HOOK_arch_g9p_gfx_520() \
   es3_trim_outputs(fs_nir, vk_state);

/* pvr_arch_g9p_gfx.c, after main line 528: the option-on insertion. */
#define PVR_G9P_ES3_HOOK_arch_g9p_gfx_528() \
   result = es3_check_targets(device, vk_state, &fs); \
   if (result != VK_SUCCESS) \
   goto out;

/* pvr_arch_g9p_gfx.c, after main line 1267: the option-on insertion. */
#define PVR_G9P_ES3_HOOK_arch_g9p_gfx_1267() \
   { \
   bool handled; \
   result = es3_clear_attachments(cmd_buffer, attachment_count, attachments, rect_count, \
   rects, &handled); \
   if (result != VK_SUCCESS || handled) \
   return result; \
   }

/* pvr_arch_g9p_gfx.c, after main line 1456: the option-on insertion. */
#define PVR_G9P_ES3_HOOK_arch_g9p_gfx_1456() \
   { \
   bool handled; \
   result = es3_job_init(cmd_buffer, hw_render, job, &handled); \
   if (result != VK_SUCCESS || handled) \
   return result; \
   }
#endif
#endif
