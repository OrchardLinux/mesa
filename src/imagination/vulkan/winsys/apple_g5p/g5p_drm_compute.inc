/* SPDX-License-Identifier: MIT */
/* Included by the G5P-owned transport; no fixture or program inspection. */
int g5p_drm_submit_compute_typed(struct g5p_drm *ws, uint32_t context,
 const struct drm_g5p_compute_pass *pass,
 const struct drm_g5p_render_resource *res, uint32_t nres,
 const struct drm_g5p_sync *syncs, uint32_t nsyncs, uint64_t *job)
{
   if (!(ws->info.capabilities & DRM_G5P_CAP_TYPED_COMPUTE)) return -EOPNOTSUPP;
   if (!nres || nres > DRM_G5P_RENDER_MAX_RESOURCES || !nsyncs || nsyncs > DRM_G5P_RENDER_MAX_SYNCS)
      return -E2BIG;
   struct drm_g5p_submit_compute_typed s = {.context=context,
      .resources=(uintptr_t)res,.resource_count=nres,.syncs=(uintptr_t)syncs,
      .sync_count=nsyncs,.pass=(uintptr_t)pass,.pass_size=sizeof(*pass)};
   int r=call(ws->fd,DRM_IOCTL_G5P_SUBMIT_COMPUTE_TYPED,&s);
   if (!r && job) *job=s.job_id;
   return r;
}
