/* SPDX-License-Identifier: MIT */
/* PR1-style hooks: each call consumes one original blank line before the next
 * line-sensitive statement. Definitions capture the call site locals. No #line. */
#ifndef PVR_G9P_ES3_ARCH_G9P_GFX_HOOKS_H
#define PVR_G9P_ES3_ARCH_G9P_GFX_HOOKS_H
#include "pvr_g9p_es3_hooks.h"
#ifdef PVR_G9P_ES3

#include "g9p_es3_isa.h"
#include "pvr_buffer.h"
#include "pvr_g9p_es3.h"

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
static void es3_pipeline_state(struct pvr_g9p_graphics_state *state, const struct g9p_shader *vs);
static VkResult es3_xfb_draw(struct pvr_cmd_buffer *cmd_buffer,
                             const struct pvr_g9p_graphics_state *state,
                             const struct pvr_g9p_draw *draw, VkPrimitiveTopology topology);
static uint64_t es3_xfb_addr(const struct pvr_cmd_buffer *cmd_buffer, unsigned binding);
static VkResult es3_ds_check(struct pvr_cmd_buffer *cmd_buffer,
                             const struct pvr_renderpass_hwsetup_render *hw);
static void es3_frag_stream_zls(const struct pvr_render_job *job, uint8_t *buf,
                                uint32_t zlsctl_at, uint32_t bgobj_at);
static void es3_draw_isp(const struct pvr_cmd_buffer *cmd_buffer,
                         const struct pvr_g9p_graphics_state *state, bool scissor,
                         uint32_t *ppp_block);
static void es3_view_state(const struct pvr_image_view *iview, uint32_t d[4]);

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

/* pvr_arch_g9p_gfx.c, after main line 562: the option-on insertion. */
#define PVR_G9P_ES3_HOOK_arch_g9p_gfx_562() \
   es3_pipeline_state(state, &vs);

/* pvr_arch_g9p_gfx.c, after main line 817: the option-on insertion. */
#define PVR_G9P_ES3_HOOK_arch_g9p_gfx_817() \
   } else if (set == G9P_XFB) { \
   addrs[k] = es3_xfb_addr(cmd_buffer, binding);

/* pvr_arch_g9p_gfx.c, after main line 1012: the option-on insertion. */
#define PVR_G9P_ES3_HOOK_arch_g9p_gfx_1012() \
   result = es3_xfb_draw(cmd_buffer, state, draw, dyn->ia.primitive_topology); \
   if (result != VK_SUCCESS) \
   return result;
/* pvr_arch_g9p_gfx.c, after main line 919: the option-on insertion. */
#define PVR_G9P_ES3_HOOK_arch_g9p_gfx_919() \
   /* -Dimagination-g9p-es3: ISP words of a draw that sets its own (es3_ds_clear()) */ \
   bool es3_isp_set; \
   uint32_t es3_isp[5];

/* pvr_arch_g9p_gfx.c, after main line 1139: the option-on insertion. */
#define PVR_G9P_ES3_HOOK_arch_g9p_gfx_1139() \
   /* ISP words with a stencil aspect (pvr_arch_g9p_es3_gfx.h) */ \
   if (dp->es3_isp_set) \
   memcpy(&ppp[blocks[nb - 1][0] + 1], dp->es3_isp, sizeof(dp->es3_isp)); \
   else \
   es3_draw_isp(cmd_buffer, state, dp->scissor >= 0, &ppp[blocks[nb - 1][0]]);

/* pvr_arch_g9p_gfx.c, after main line 1267: the option-on insertion. */
#define PVR_G9P_ES3_HOOK_arch_g9p_gfx_1267() \
   { \
   bool handled; \
   result = es3_clear_attachments(cmd_buffer, attachment_count, attachments, rect_count, \
   rects, &handled); \
   if (result != VK_SUCCESS || handled) \
   return result; \
   }

/* pvr_arch_g9p_gfx.c, after main line 1376: the option-on insertion. */
#define PVR_G9P_ES3_HOOK_arch_g9p_gfx_1376() \
   es3_view_state(iview, d);

/* pvr_arch_g9p_gfx.c, after main line 1456: the option-on insertion. */
#define PVR_G9P_ES3_HOOK_arch_g9p_gfx_1456() \
   result = es3_ds_check(cmd_buffer, hw_render); \
   if (result != VK_SUCCESS) \
   return result; \
   { \
   bool handled; \
   result = es3_job_init(cmd_buffer, hw_render, job, &handled); \
   if (result != VK_SUCCESS || handled) \
   return result; \
   }
#endif
#endif
