/* SPDX-License-Identifier: MIT */
/* Included by pvr_g5p_winsys.c after the render-context helpers.
 * Track compute's last fence in the existing idle/VM-edit list, without
 * modifying the speed lane's arena/RAM policy. */
struct g5p_ws_compute_ctx {
   struct pvr_winsys_compute_ctx base;
   struct g5p_ws_context *tracked;
};
static VkResult g5p_ws_compute_ctx_create(struct pvr_winsys *base,
 const struct pvr_winsys_compute_ctx_create_info *info, const struct pvr_device_info *dev_info,
 struct pvr_winsys_compute_ctx **out)
{
   (void)info;(void)dev_info;
   struct g5p_ws_compute_ctx *c=vk_zalloc(base->alloc,sizeof(*c),8,VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if(!c)return vk_error(NULL,VK_ERROR_OUT_OF_HOST_MEMORY);
   c->base.ws=base;
   c->tracked=g5p_ws_context_get(to_ws(base));
   if(!c->tracked){vk_free(base->alloc,c);return vk_error(NULL,VK_ERROR_OUT_OF_HOST_MEMORY);}
   *out=&c->base;return VK_SUCCESS;
}
static void g5p_ws_compute_ctx_destroy(struct pvr_winsys_compute_ctx *base)
{
   struct g5p_ws_compute_ctx *c=container_of(base,struct g5p_ws_compute_ctx,base);
   g5p_ws_context_put(to_ws(base->ws),c->tracked);
   vk_free(base->ws->alloc,c);
}
static VkResult g5p_ws_compute_submit(const struct pvr_winsys_compute_ctx *base,
 const struct pvr_winsys_compute_submit_info *info, const struct pvr_device_info *dev_info,
 struct vk_sync *signal)
{
   (void)dev_info;
   struct g5p_ws_compute_ctx *c=container_of(base,struct g5p_ws_compute_ctx,base);
   struct g5p_ws *ws=to_ws(base->ws);
   if(!(ws->drm.info.capabilities&DRM_G5P_CAP_TYPED_COMPUTE))
      return g5p_ws_refuse(VK_ERROR_FEATURE_NOT_PRESENT,"kernel has no typed compute submit");
   if(info->fw_stream_len!=sizeof(struct pvr_g5p_compute_stream))
      return g5p_ws_refuse(VK_ERROR_FEATURE_NOT_PRESENT,"compute stream ABI mismatch");
   struct pvr_g5p_compute_stream stream;memcpy(&stream,info->fw_stream,sizeof(stream));
   struct drm_g5p_compute_pass pass={.size=sizeof(pass),.cdm_va=stream.cdm_va,
      .cdm_size=stream.cdm_size,.cdm_end=stream.cdm_end,.state_tail_va=stream.tail_va,
      .config={0x40200003,0}};
   for(unsigned i=0;i<4;i++)pass.state_va[i]=stream.state_va+4*i;
   /* Serial execution: generated HALT context tasks, no captured save/restore window. */
   pass.store=(struct drm_g5p_compute_pds){stream.halt_code,stream.halt_data,0,0};
   pass.resume=pass.store;pass.terminate=pass.store;
   struct drm_g5p_render_resource res[DRM_G5P_RENDER_MAX_RESOURCES];unsigned nres=0;
   simple_mtx_lock(&ws->lock);
   int r=0;
   r=g5p_ws_context_prepare(ws,c->tracked);
   if(r){simple_mtx_unlock(&ws->lock);return g5p_ws_refuse(VK_ERROR_DEVICE_LOST,"compute CREATE_CONTEXT: %s",strerror(-r));}
   if(ws->chunk_count+ws->own_count>ARRAY_SIZE(res)) {
      simple_mtx_unlock(&ws->lock);return vk_error(NULL,VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }
   /* Conservative whole-resource visibility. No render-only range narrowing. */
   for(unsigned i=0;i<ws->chunk_count;i++)res[nres++]=(struct drm_g5p_render_resource){
      .handle=ws->chunks[i].handle,.va=ws->chunks[i].va,.size=ws->chunks[i].size,
      .kind=DRM_G5P_RENDER_ACCESS_READ|DRM_G5P_RENDER_ACCESS_WRITE|DRM_G5P_RENDER_ACCESS_CACHE};
   list_for_each_entry(struct g5p_ws_bo,b,&ws->own_bos,link)res[nres++]=(struct drm_g5p_render_resource){
      .handle=b->kbo.handle,.va=b->va,.size=b->kbo.size,
      .kind=DRM_G5P_RENDER_ACCESS_READ|DRM_G5P_RENDER_ACCESS_WRITE|DRM_G5P_RENDER_ACCESS_CACHE|
         (b->kbo.shared?DRM_G5P_RENDER_ACCESS_IMPLICIT:0)};
   struct drm_g5p_sync syncs[2];unsigned n=0;
   if(info->wait)syncs[n++]=(struct drm_g5p_sync){.handle=vk_sync_as_drm_syncobj(info->wait)->syncobj};
   uint32_t local_signal=0;
   if(!signal && drmSyncobjCreate(ws->base.render_fd,0,&local_signal)) {
      simple_mtx_unlock(&ws->lock);return vk_error(NULL,VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   uint32_t sig=signal?vk_sync_as_drm_syncobj(signal)->syncobj:local_signal;
   syncs[n++]=(struct drm_g5p_sync){.handle=sig,.flags=DRM_G5P_SYNC_SIGNAL};
   uint64_t job=0;
   r=g5p_drm_submit_compute_typed(&ws->drm,c->tracked->context,&pass,res,nres,syncs,n,&job);
   if(!r) {
      if(!r && drmSyncobjTransfer(ws->base.render_fd,c->tracked->last,0,sig,0,0))r=-errno;
   }
   if(local_signal)drmSyncobjDestroy(ws->base.render_fd,local_signal);
   simple_mtx_unlock(&ws->lock);
   if(getenv("PVR_G5P_WS_LOG"))fprintf(stderr,"apple_g5p compute: ctx=%u job=%"PRIu64" cdm=%#"PRIx64"+%u resources=%u ret=%d\n",c->tracked->context,job,stream.cdm_va,stream.cdm_end,nres,r);
   return r?g5p_ws_refuse(VK_ERROR_DEVICE_LOST,"SUBMIT_COMPUTE_TYPED: %s",strerror(-r)):VK_SUCCESS;
}
