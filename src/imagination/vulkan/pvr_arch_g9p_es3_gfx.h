/* SPDX-License-Identifier: MIT */
/*
 * -Dimagination-g9p-es3: the OpenGL ES 3.0 enablers of G9P graphics.
 * Included by pvr_arch_g9p_gfx.c only, at its end and under PVR_G9P_ES3, so
 * that it uses that file's helpers (upload_kick(), pack_image_state(),
 * emit_draw(), ...) instead of copies; the hooks there call these
 * functions, each behind PVR_G9P_ES3.
 *
 * Ported from our Apple A9 (G5P) driver; where the A10 differs, or may
 * differ, from the A9, the comment says so. Nothing here has run on an A10.
 *
 * Several colour targets (up to four 8-bit RGBA ones). On the A9 (mrt.md,
 * hardware-proven for two and four targets, pixel-exact):
 *  - the fragment shader writes target t to output halves 4t..4t+3 and one
 *    emit writes all 2n output registers (g9p_compile.c end_targets());
 *  - the end-of-tile DOUTD moves eight common-store dwords per target: the
 *    target's image state (seven dwords, as for one target), its
 *    output-half offset 4t in dword 6 and 0xffffffff in dword 7 between
 *    targets; the USC program puts each offset in a register and emits one
 *    PBE write per target (g9p_es3_pbe_emit());
 *  - the background DOUTD moves four f32 clear colours per target to
 *    common-store dwords 4t..4t+3 and the clear program converts them into
 *    the output halves (g9p_gfx_device_init()'s instruction for every half);
 *  - EVENT_PIXEL_PDS_INFO grows from 0x402 to 0x802 with four targets;
 *  - the pixel output width follows the output registers: 2, 4, 8.
 * The A10 differences:
 *  - the output width registers. On the A9 four words follow the width
 *    (A9 notes, mrt.md: registers W3c8, W3cc, the six per-core W108..W130
 *    and W6b0; 2 registers: 8, f8, 3c3, 2; 4: 4, f9, 1d3, 4; 8: 2, fa, e3,
 *    8). The A10's one-target job already carries the A9's 4-register
 *    values in the two of them its stream holds: 0x1d3 in each of
 *    cluster 1's words (0x11080) and 0xf9 in 0x4070. So two targets need no
 *    change there, and four take the 8-register values 0xe3 and 0xfa. The
 *    other two words (A9 W3c8 and W6b0) are not in the A10 stream; if four
 *    targets come out wrong on the A10, they are the first suspects, and
 *    they live in the kernel's command template, not here;
 *  - the end-of-tile offsets are immediates (g9p_mov_imm(), the A10's own
 *    form in the one-target program) where Apple's A9 code loads them
 *    from dword 6 of each target's state; the values are the same;
 *  - the load program samples each target with the A10's LOAD sequence
 *    (iPadOS's `load` capture on the A10), the coordinates kept above the
 *    output registers. Apple's A9 driver reloads several targets with its
 *    own restore program, which we never decoded.
 * Refused by name: three targets (Apple's A9 compiler was only observed for
 * 1, 2 and 4), a gap in the targets, blending, logic ops or partial write
 * masks with several targets, mixed or non-32-bit target formats, a mix of
 * clear and load ops, merged subpasses, and clears of some targets only.
 */

/* --- Several colour targets: device programs ------------------------------ */

/* index into the G9P_ES3 program arrays by the target count (2 or 4) */
static unsigned
es3_mrt_index(unsigned n)
{
   return n == 4;
}

static VkResult
es3_gfx_device_init(struct pvr_device *device, struct g9p_code *c)
{
   struct pvr_g9p_gfx_programs *p = &device->g9p.gfx;
   VkResult result = VK_SUCCESS;

   for (unsigned i = 0; i < 2 && result == VK_SUCCESS; i++) {
      const unsigned n = i ? 4 : 2;

      g9p_es3_clear_program(c, n);
      result = upload_usc(device, c, &p->es3_clear_usc[i]);
      if (result == VK_SUCCESS)
         result = upload_kick(device, p->es3_clear_usc[i], 8, 0, &p->es3_clear_kick[i]);
      if (result != VK_SUCCESS)
         break;
      g9p_es3_load_program(c, n);
      result = upload_usc(device, c, &p->es3_load_usc[i]);
      if (result == VK_SUCCESS)
         result = upload_kick(device, p->es3_load_usc[i], 2 * n + 2, 0, &p->es3_load_kick[i]);
   }
   /* end of tile: every non-empty set of stored targets of four */
   for (unsigned mask = 1; mask < 16 && result == VK_SUCCESS; mask++) {
      g9p_es3_eot_program(c, mask);
      result = upload_usc(device, c, &p->es3_eot_usc[mask]);
   }
   return result;
}

static void
es3_gfx_device_finish(struct pvr_device *device)
{
   struct pvr_g9p_gfx_programs *p = &device->g9p.gfx;

   for (unsigned i = 0; i < 2; i++) {
      if (p->es3_clear_kick[i].pvr_bo)
         pvr_bo_suballoc_free(p->es3_clear_kick[i].pvr_bo);
      if (p->es3_load_kick[i].pvr_bo)
         pvr_bo_suballoc_free(p->es3_load_kick[i].pvr_bo);
      pvr_bo_suballoc_free(p->es3_clear_usc[i]);
      pvr_bo_suballoc_free(p->es3_load_usc[i]);
   }
   for (unsigned mask = 1; mask < 16; mask++)
      pvr_bo_suballoc_free(p->es3_eot_usc[mask]);
}

/* --- Several colour targets: pipelines ------------------------------------ */

/*
 * Before the fragment shader is compiled: outputs past the pipeline's
 * colour attachments are dropped, as GL drops them (they would otherwise
 * count as targets)
 */
static void
es3_trim_outputs(nir_shader *fs, const struct vk_graphics_pipeline_state *vk_state)
{
   const unsigned n = vk_state->rp ? vk_state->rp->color_attachment_count : 0;
   bool progress = false;

   if (!n)
      return;
   nir_foreach_shader_out_variable (var, fs) {
      if (var->data.location >= (int)FRAG_RESULT_DATA0 + (int)n &&
          var->data.location < (int)FRAG_RESULT_DATA0 + 8) {
         var->data.mode = nir_var_shader_temp;
         progress = true;
      }
   }
   if (progress) {
      nir_fixup_deref_modes(fs);
      NIR_PASS(_, fs, nir_lower_vars_to_ssa);
      NIR_PASS(_, fs, nir_opt_dce);
   }
}

/* The formats the end of tile stores for several targets: 32-bit RGBA ones */
static bool
es3_mrt_format(VkFormat format)
{
   return vk_format_get_blocksizebits(format) == 32 && g9p_image_format(format) &&
          !vk_format_is_int(format);
}

/* After both shaders are compiled: what several colour targets cannot do yet */
static VkResult
es3_check_targets(struct pvr_device *device, const struct vk_graphics_pipeline_state *vk_state,
                  const struct g9p_shader *fs)
{
   const struct vk_render_pass_state *rp = vk_state->rp;
   const struct vk_color_blend_state *cb = vk_state->cb;
   const unsigned n = rp ? rp->color_attachment_count : 0;

   if (n <= 1 && fs->num_targets <= 1)
      return VK_SUCCESS;
   if (n == 3 || n > G9P_ES3_MAX_TARGETS)
      return g9p_error(device, "G9P: %u colour attachments (only 1, 2 or 4 are characterized)", n);
   if (fs->uses_fbfetch || fs->native_blend)
      return g9p_error(device, "G9P: tile reads with several colour targets");
   for (unsigned t = 0; t < n; t++) {
      if (rp->color_attachment_formats[t] == VK_FORMAT_UNDEFINED)
         return g9p_error(device, "G9P: colour attachment %u unused with several targets", t);
      if (rp->color_attachment_formats[t] != rp->color_attachment_formats[0] ||
          !es3_mrt_format(rp->color_attachment_formats[t]))
         return g9p_error(device, "G9P: colour attachment %u format %s with several targets "
                          "(8-bit RGBA ones, all alike, only)",
                          t, vk_Format_to_str(rp->color_attachment_formats[t]));
      if (cb && t < cb->attachment_count &&
          (cb->attachments[t].blend_enable || (cb->attachments[t].write_mask & 0xf) != 0xf))
         return g9p_error(device, "G9P: blending or a write mask with several colour targets");
   }
   if (cb && cb->logic_op_enable)
      return g9p_error(device, "G9P: a logic op with several colour targets");
   return VK_SUCCESS;
}

/* --- Several colour targets: render jobs ---------------------------------- */

/* The number of colour targets of a render's (only) subpass */
static unsigned
es3_render_targets(const struct pvr_renderpass_hwsetup_render *hw_render)
{
   if (!hw_render || !hw_render->subpass_count)
      return 0;
   return hw_render->subpasses[0].setup.num_render_targets;
}

static bool
es3_same_resource(const struct usc_mrt_resource *a, const struct usc_mrt_resource *b)
{
   if (a->type != b->type)
      return false;
   if (a->type == USC_MRT_RESOURCE_TYPE_OUTPUT_REG)
      return a->reg.output_reg == b->reg.output_reg && a->reg.offset == b->reg.offset;
   return a->mem.tile_buffer == b->mem.tile_buffer && a->mem.offset_dw == b->mem.offset_dw;
}

/*
 * Colour target @t's attachment in the render's attachments: the surface
 * whose on-chip storage is the one the subpass gives location t; -1 when
 * nothing stores it at the end of the render
 */
static int
es3_eot_attachment(const struct pvr_renderpass_hwsetup_render *hw_render, unsigned t)
{
   const struct usc_mrt_resource *res = &hw_render->subpasses[0].setup.mrt_resources[t];

   for (unsigned i = 0; i < hw_render->eot_surface_count; i++) {
      const struct pvr_renderpass_hwsetup_eot_surface *s = &hw_render->eot_surfaces[i];

      if (es3_same_resource(&hw_render->eot_setup.mrt_resources[s->mrt_idx], res))
         return s->attachment_idx;
   }
   return -1;
}

/* Colour target @t's load or clear at the start of the render, or NULL */
static const struct pvr_renderpass_colorinit *
es3_color_init(const struct pvr_renderpass_hwsetup_render *hw_render, unsigned t)
{
   const struct usc_mrt_resource *res = &hw_render->subpasses[0].setup.mrt_resources[t];

   for (unsigned i = 0; i < hw_render->color_init_count; i++) {
      if (i < hw_render->init_setup.num_render_targets &&
          es3_same_resource(&hw_render->init_setup.mrt_resources[i], res))
         return &hw_render->color_init[i];
   }
   return NULL;
}

/*
 * g9p_job_init() for a render with several colour targets; *@handled false
 * leaves it to the one-target path
 */
static VkResult
es3_job_init(struct pvr_cmd_buffer *cmd_buffer,
             const struct pvr_renderpass_hwsetup_render *hw_render, struct pvr_render_job *job,
             bool *handled)
{
   struct pvr_device *device = cmd_buffer->device;
   const struct pvr_g9p_gfx_programs *p = &device->g9p.gfx;
   const struct pvr_render_pass_info *info = &cmd_buffer->state.render_pass_info;
   const unsigned n = es3_render_targets(hw_render);
   uint32_t image[8 * G9P_ES3_MAX_TARGETS] = { 0 };
   uint32_t bg[16 * G9P_ES3_MAX_TARGETS] = { 0 };
   int attachment[G9P_ES3_MAX_TARGETS];
   unsigned stored = 0, m = 0, loads = 0, clears = 0, bg_dwords;
   struct pvr_suballoc_bo *bo;
   struct pvr_pds_upload up;
   VkResult result;

   *handled = n > 1;
   if (n <= 1)
      return VK_SUCCESS;
   if (n == 3 || n > G9P_ES3_MAX_TARGETS || hw_render->subpass_count != 1)
      return vk_errorf(cmd_buffer, VK_ERROR_FEATURE_NOT_PRESENT,
                       "G9P: a render of %u colour targets in %u subpasses", n,
                       hw_render->subpass_count);

   memset(&job->g9p, 0, sizeof(job->g9p));
   job->g9p.enabled = true;
   job->g9p.num_targets = n;

   /* end of tile: the stored targets' image states, eight dwords each */
   for (unsigned t = 0; t < n; t++) {
      const struct pvr_image_view *iview;
      const struct pvr_image *img;
      const struct pvr_image_plane *plane;
      uint64_t addr;
      uint32_t pitch, *w = &image[8 * m];
      bool linear;

      if (hw_render->subpasses[0].setup.mrt_resources[t].type == USC_MRT_RESOURCE_TYPE_INVALID)
         return vk_errorf(cmd_buffer, VK_ERROR_FEATURE_NOT_PRESENT,
                          "G9P: colour target %u unused with several targets", t);
      attachment[t] = es3_eot_attachment(hw_render, t);
      if (attachment[t] < 0)
         continue;
      iview = info->attachments[attachment[t]];
      img = pvr_image_view_get_image(iview);
      plane = pvr_single_plane_const(img);
      addr = img->dev_addr.addr + plane->offset +
             plane->mip_levels[iview->vk.base_mip_level].offset +
             iview->vk.base_array_layer * plane->layer_size;
      linear = img->memlayout == PVR_MEMLAYOUT_LINEAR;
      pitch = plane->mip_levels[iview->vk.base_mip_level].pitch;
      if (!es3_mrt_format(iview->vk.format))
         return vk_errorf(cmd_buffer, VK_ERROR_FEATURE_NOT_PRESENT,
                          "G9P: colour target %u format %s with several targets", t,
                          vk_Format_to_str(iview->vk.format));
      pack_image_state(w, addr, iview->vk.extent.width, iview->vk.extent.height, linear, pitch,
                       iview->vk.format);
      w[6] = 4 * t;
      if (!m) {
         job->g9p.target_addr = addr;
         job->g9p.target_pitch = pitch;
         job->g9p.target_linear = linear;
         job->g9p.width = iview->vk.extent.width;
         job->g9p.height = iview->vk.extent.height;
      }
      stored |= BITFIELD_BIT(t);
      m++;
   }
   if (!stored) {
      /* nothing stored: the one-target path's empty end of tile */
      *handled = false;
      return VK_SUCCESS;
   }
   for (unsigned k = 0; k + 1 < m; k++)
      image[8 * k + 7] = 0xffffffff;
   result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, image, 4 * (8 * m - 1), &bo);
   if (result != VK_SUCCESS)
      return result;
   {
      const uint32_t data[8] = { bo->dev_addr.addr, bo->dev_addr.addr >> 32,
                                 doutd_ctrl(0, 8 * m - 1, true), 0,
                                 usc_offset(device, p->es3_eot_usc[stored]), usc_temps(8), 0, 0 };

      result = pvr_arch_cmd_buffer_upload_pds(cmd_buffer, data, ARRAY_SIZE(data), 16, NULL,
                                              0, 0, 64, &up);
      if (result != VK_SUCCESS)
         return result;
   }
   job->view_state.view[0].pds_pixel_event_data_offset = up.data_offset;
   job->g9p.eot_code = p->eot.code_offset;
   /*
    * EVENT_PIXEL_PDS_INFO: usc_sr_size twice DIV_ROUND_UP(dwords, 16), as
    * Apple's A9 driver sets it (0x402 for 7 and 15 dwords, 0x802 for 31)
    */
   job->g9p.eot_info = 2 * DIV_ROUND_UP(8 * m - 1, 16) << 9 | 2;

   /*
    * Background object: every target cleared (don't care: transparent
    * black, as for one target) or every target loaded (a don't-care one is
    * loaded too)
    */
   for (unsigned t = 0; t < n; t++) {
      const struct pvr_renderpass_colorinit *ci = es3_color_init(hw_render, t);

      if (ci && ci->op == VK_ATTACHMENT_LOAD_OP_LOAD)
         loads |= BITFIELD_BIT(t);
      else if (ci && ci->op == VK_ATTACHMENT_LOAD_OP_CLEAR)
         clears |= BITFIELD_BIT(t);
   }
   if (loads && clears)
      return vk_errorf(cmd_buffer, VK_ERROR_FEATURE_NOT_PRESENT,
                       "G9P: colour targets %#x loaded and %#x cleared in one render", loads,
                       clears);
   job->g9p.load = loads != 0;
   if (loads) {
      for (unsigned t = 0; t < n; t++) {
         const struct pvr_renderpass_colorinit *ci = es3_color_init(hw_render, t);
         const struct pvr_image_view *iview;
         uint64_t tex[2];

         if (!ci)
            return vk_errorf(cmd_buffer, VK_ERROR_FEATURE_NOT_PRESENT,
                             "G9P: colour target %u has no storage to load into", t);
         iview = info->attachments[ci->index];
         PVR_PER_ARCH(g9p_pack_image_view)(iview, tex);
         memcpy(&bg[16 * t], tex, sizeof(tex));
         bg[16 * t + 8] = 0x000e0000;
         bg[16 * t + 9] = 0x00c00780;
         bg[16 * t + 12] = fui(1.0f / (16 * iview->vk.extent.width));
         bg[16 * t + 13] = fui(1.0f / (16 * iview->vk.extent.height));
      }
      bg_dwords = 16 * n;
   } else {
      for (unsigned t = 0; t < n; t++) {
         const struct pvr_renderpass_colorinit *ci = es3_color_init(hw_render, t);

         if (ci && ci->op == VK_ATTACHMENT_LOAD_OP_CLEAR && ci->index < info->clear_value_count)
            memcpy(&bg[4 * t], info->clear_values[ci->index].color.float32, 4 * sizeof(float));
      }
      bg_dwords = 4 * n;
   }
   {
      const struct pvr_renderpass_colorinit *ci0 = es3_color_init(hw_render, 0);

      job->g9p.load_op = ci0 ? (int)ci0->op : G9P_LOAD_OP_NONE;
   }
   result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, bg, 4 * bg_dwords, &bo);
   if (result != VK_SUCCESS)
      return result;
   {
      const uint32_t data[4] = { bo->dev_addr.addr, bo->dev_addr.addr >> 32,
                                 doutd_ctrl(0, bg_dwords, true), 0 };

      result = pvr_arch_cmd_buffer_upload_pds(cmd_buffer, data, ARRAY_SIZE(data), 16, NULL,
                                              0, 0, 64, &up);
      if (result != VK_SUCCESS)
         return result;
   }
   job->g9p.bgnd[0] = (uint64_t)p->dma.code_offset << 32 |
                      (loads ? p->es3_load_kick[es3_mrt_index(n)].data_offset
                             : p->es3_clear_kick[es3_mrt_index(n)].data_offset);
   job->g9p.bgnd[1] = 0;
   job->g9p.bgnd[2] = up.data_offset;
   /*
    * PDS_BGRND3_SIZEINFO: usc_sharedsize in 16-dword units (Rogue's field),
    * one up to 16 dwords as for one target (and on the A9 for up to four
    * cleared targets); the loads of two and four targets take 32 and 64
    * dwords, which no capture shows
    */
   job->g9p.bgnd[3] = (uint64_t)DIV_ROUND_UP(bg_dwords, 16) << 55 | 0x00800000ull;
   dump32("es3 image states", 0, image, 8 * m);
   return VK_SUCCESS;
}

/*
 * The fragment stream's words that follow the targets (offsets of the
 * words g9p_frag_stream() put): EVENT_PIXEL_PDS_INFO (es3_job_init()),
 * cluster 1's six words and 0x4070 (see the top: the A10's one-target
 * values are the A9's 4-register ones; four targets take the 8-register
 * ones)
 */
static void
es3_frag_stream_patch(const struct pvr_render_job *job, uint8_t *buf, uint32_t info_at,
                      uint32_t cluster1_at, uint32_t r4070_at)
{
   if (job->g9p.num_targets <= 1)
      return;
   *(uint32_t *)(buf + info_at) = job->g9p.eot_info;
   if (job->g9p.num_targets != 4)
      return;
   for (unsigned i = 0; i < 6; i++)
      *(uint64_t *)(buf + cluster1_at + 8 * i) = 0xe3;
   *(uint32_t *)(buf + r4070_at) = 0xfa;
}

/*
 * vkCmdClearAttachments in a render with several colour targets: one draw
 * of the n-target clear program over the rectangles, every target's
 * colour by the clear's DMA. A target the clear leaves alone would be
 * overwritten as well, so clears of some targets only are refused.
 */
static VkResult
es3_clear_attachments(struct pvr_cmd_buffer *cmd_buffer, uint32_t attachment_count,
                      const VkClearAttachment *attachments, uint32_t rect_count,
                      const VkClearRect *rects, bool *handled)
{
   const struct pvr_g9p_gfx_programs *p = &cmd_buffer->device->g9p.gfx;
   const struct pvr_render_pass_info *info = &cmd_buffer->state.render_pass_info;
   const unsigned n = es3_render_targets(info->hw_render);
   const float w = info->render_area.offset.x + info->render_area.extent.width;
   const float h = info->render_area.offset.y + info->render_area.extent.height;
   float colours[4 * G9P_ES3_MAX_TARGETS] = { 0 };
   unsigned cleared = 0;
   struct pvr_g9p_graphics_state state = {
      .vs_kick = p->clear_vs_kick,
      .vs_outputs = 4,
      .vs_regs = 8,
      .num_buffers = 1,
   };
   struct pvr_pds_upload dma, colour_dma;
   struct g9p_draw_programs dp = {
      .dma = &dma, .tex = &colour_dma,
      .vp = { w / 2, w / 2, h / 2, h / 2, 0, 1 },
      .scissor = -1,
   };
   struct pvr_suballoc_bo *verts_bo, *table_bo, *colour_bo;
   float *verts;
   VkResult result;

   *handled = n > 1;
   if (n <= 1)
      return VK_SUCCESS;
   for (uint32_t i = 0; i < attachment_count; i++) {
      const unsigned t = attachments[i].colorAttachment;

      if (!(attachments[i].aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) || t >= n)
         continue;
      memcpy(&colours[4 * t], attachments[i].clearValue.color.float32, 4 * sizeof(float));
      cleared |= BITFIELD_BIT(t);
   }
   if (!cleared || !rect_count)
      return VK_SUCCESS;
   if (cleared != BITFIELD_MASK(n))
      return vk_errorf(cmd_buffer, VK_ERROR_FEATURE_NOT_PRESENT,
                       "G9P: a clear of colour targets %#x of %u", cleared, n);

   state.fs_kick = p->es3_clear_kick[es3_mrt_index(n)];
   dp.fs_common_dwords = 4 * n;
   result = pvr_arch_cmd_buffer_start_sub_cmd(cmd_buffer, PVR_SUB_CMD_TYPE_GRAPHICS);
   if (result != VK_SUCCESS)
      return result;
   verts = malloc(rect_count * 12 * sizeof(float));
   if (!verts)
      return vk_error(cmd_buffer, VK_ERROR_OUT_OF_HOST_MEMORY);
   for (uint32_t r = 0; r < rect_count; r++) {
      const VkRect2D *rc = &rects[r].rect;
      const float x0 = 2 * rc->offset.x / w - 1, y0 = 2 * rc->offset.y / h - 1;
      const float x1 = 2 * (rc->offset.x + rc->extent.width) / w - 1;
      const float y1 = 2 * (rc->offset.y + rc->extent.height) / h - 1;
      const float v[12] = { x0, y0, x1, y0, x0, y1, x1, y0, x1, y1, x0, y1 };

      memcpy(&verts[12 * r], v, sizeof(v));
   }
   result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, verts, rect_count * 12 * sizeof(float),
                                               &verts_bo);
   free(verts);
   if (result != VK_SUCCESS)
      return result;
   result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, &verts_bo->dev_addr.addr, 8,
                                               &table_bo);
   if (result != VK_SUCCESS)
      return result;
   {
      const uint32_t data[4] = { table_bo->dev_addr.addr, table_bo->dev_addr.addr >> 32,
                                 doutd_ctrl(0, 2, true), 0 };

      result = pvr_arch_cmd_buffer_upload_pds(cmd_buffer, data, ARRAY_SIZE(data), 16, NULL, 0,
                                              0, 64, &dma);
      if (result != VK_SUCCESS)
         return result;
   }
   result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, colours, 4 * 4 * n, &colour_bo);
   if (result != VK_SUCCESS)
      return result;
   {
      const uint32_t data[4] = { colour_bo->dev_addr.addr, colour_bo->dev_addr.addr >> 32,
                                 doutd_ctrl(0, 4 * n, true), 0 };

      result = pvr_arch_cmd_buffer_upload_pds(cmd_buffer, data, ARRAY_SIZE(data), 16, NULL, 0,
                                              0, 64, &colour_dma);
      if (result != VK_SUCCESS)
         return result;
      colour_dma.code_offset = p->dma.code_offset;
   }
   return emit_draw(cmd_buffer, &state, &dp,
                    &(const struct pvr_g9p_draw){ .count = 6 * rect_count, .instances = 1 },
                    VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
}

/* --- Transform feedback ---------------------------------------------------- */

/*
 * The vertex shader stores its captured outputs itself (g9p_compile.c
 * lower_xfb(), our A9 driver's method), so what is left here is where each
 * draw's records go and how many fit. As on the A9: non-indexed list
 * draws, stream 0, whole primitives only, the buffer with the least room
 * decides. Where our A9 driver resolves the counters when the command
 * buffer executes (its queue runs on the CPU between jobs), this one
 * resolves them while recording: Begin reads a counter buffer through the
 * CPU mapping and End writes it, which is right when command buffers
 * execute in the order they are recorded, as Zink's do. Queries and
 * vkCmdDrawIndirectByteCountEXT are not built.
 */

static unsigned
es3_xfb_vertices_per_primitive(VkPrimitiveTopology t)
{
   switch (t) {
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
      return 1;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
      return 2;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
      return 3;
   default:
      return 0;
   }
}

static void
es3_pipeline_state(struct pvr_g9p_graphics_state *state, const struct g9p_shader *vs)
{
   state->es3_xfb_buffers = vs->xfb_buffers;
   memcpy(state->es3_xfb_stride, vs->xfb_stride, sizeof(state->es3_xfb_stride));
}

/*
 * Before a draw's vertex uniforms: its capture parameters and record
 * bases, and the bytes the capture adds to each buffer
 */
static VkResult
es3_xfb_draw(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_g9p_graphics_state *state,
             const struct pvr_g9p_draw *draw, VkPrimitiveTopology topology)
{
   const struct pvr_cmd_buffer_xfb_state *xs = &cmd_buffer->state.xfb_state;
   struct pvr_g9p_es3_xfb_state *x = &cmd_buffer->state.g9p_es3_xfb;
   const unsigned vpp = es3_xfb_vertices_per_primitive(topology);
   uint32_t params[4] = { 0 };
   struct pvr_suballoc_bo *bo;
   uint64_t records = 0;
   VkResult result;

   if (!state->es3_xfb_buffers)
      return VK_SUCCESS;
   if (x->active) {
      uint64_t fit = UINT64_MAX;

      if (draw->indexed || !vpp)
         return vk_errorf(cmd_buffer, VK_ERROR_FEATURE_NOT_PRESENT,
                          "G9P: transform feedback of an indexed draw or of topology %u "
                          "(non-indexed lists only)",
                          topology);
      if (draw->count % vpp && MAX2(draw->instances, 1) > 1)
         return vk_errorf(cmd_buffer, VK_ERROR_FEATURE_NOT_PRESENT,
                          "G9P: transform feedback of instances with an incomplete primitive");
      params[G9P_XFB_PARAM_VERTICES] = draw->count - draw->count % vpp;
      u_foreach_bit (b, state->es3_xfb_buffers) {
         const struct pvr_xfb_binding *bind = &xs->bindings[b];
         const uint64_t stride = state->es3_xfb_stride[b];
         const uint64_t room = bind->buffer_size > x->written[b]
                                  ? (bind->buffer_size - x->written[b]) / stride
                                  : 0;

         if (!bind->buffer_size)
            return vk_errorf(cmd_buffer, VK_ERROR_FEATURE_NOT_PRESENT,
                             "G9P: transform feedback buffer %u not bound", b);
         fit = MIN2(fit, room / vpp * vpp);
      }
      records = MIN2((uint64_t)params[G9P_XFB_PARAM_VERTICES] * MAX2(draw->instances, 1), fit);
   }
   params[G9P_XFB_PARAM_LIMIT] = records;
   params[G9P_XFB_PARAM_FIRST] = draw->offset;
   result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, params, sizeof(params), &bo);
   if (result != VK_SUCCESS)
      return result;
   x->params_addr = bo->dev_addr.addr;
   u_foreach_bit (b, state->es3_xfb_buffers) {
      x->base[b] = xs->bindings[b].buffer_addr.addr + x->written[b];
      x->written[b] += records * state->es3_xfb_stride[b];
   }
   return VK_SUCCESS;
}

/* buffer_addrs() of a G9P_XFB slot */
static uint64_t
es3_xfb_addr(const struct pvr_cmd_buffer *cmd_buffer, unsigned binding)
{
   const struct pvr_g9p_es3_xfb_state *x = &cmd_buffer->state.g9p_es3_xfb;

   if (binding == G9P_XFB_PARAMS)
      return x->params_addr;
   return binding < PVR_XFB_MAX_BUFFERS ? x->base[binding] : 0;
}

/* A counter buffer's 32-bit byte count, through the CPU mapping */
static uint32_t *
es3_xfb_counter(struct pvr_cmd_buffer *cmd_buffer, VkBuffer handle, VkDeviceSize offset)
{
   VK_FROM_HANDLE(pvr_buffer, buffer, handle);

   return (uint32_t *)PVR_PER_ARCH(g9p_cpu_addr)(cmd_buffer->device,
                                                 buffer->dev_addr.addr + offset);
}

void
PVR_PER_ARCH(g9p_es3_xfb_begin)(struct pvr_cmd_buffer *cmd_buffer, uint32_t first,
                                uint32_t count, const VkBuffer *counters,
                                const VkDeviceSize *offsets)
{
   struct pvr_g9p_es3_xfb_state *x = &cmd_buffer->state.g9p_es3_xfb;

   x->active = true;
   memset(x->written, 0, sizeof(x->written));
   for (uint32_t j = 0; counters && j < count; j++) {
      uint32_t *c;

      if (first + j >= PVR_XFB_MAX_BUFFERS || counters[j] == VK_NULL_HANDLE)
         continue;
      c = es3_xfb_counter(cmd_buffer, counters[j], offsets ? offsets[j] : 0);
      if (!c) {
         vk_command_buffer_set_error(&cmd_buffer->vk,
                                     vk_errorf(cmd_buffer, VK_ERROR_FEATURE_NOT_PRESENT,
                                               "G9P: transform feedback counter not mapped"));
         return;
      }
      x->written[first + j] = *c;
   }
}

void
PVR_PER_ARCH(g9p_es3_xfb_end)(struct pvr_cmd_buffer *cmd_buffer, uint32_t first, uint32_t count,
                              const VkBuffer *counters, const VkDeviceSize *offsets)
{
   struct pvr_g9p_es3_xfb_state *x = &cmd_buffer->state.g9p_es3_xfb;

   x->active = false;
   for (uint32_t j = 0; counters && j < count; j++) {
      uint32_t *c;

      if (first + j >= PVR_XFB_MAX_BUFFERS || counters[j] == VK_NULL_HANDLE)
         continue;
      c = es3_xfb_counter(cmd_buffer, counters[j], offsets ? offsets[j] : 0);
      if (!c) {
         vk_command_buffer_set_error(&cmd_buffer->vk,
                                     vk_errorf(cmd_buffer, VK_ERROR_FEATURE_NOT_PRESENT,
                                               "G9P: transform feedback counter not mapped"));
         return;
      }
      *c = x->written[first + j];
   }
}
