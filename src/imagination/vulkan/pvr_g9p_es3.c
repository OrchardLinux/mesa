/* SPDX-License-Identifier: MIT */
/*
 * Apple G9P (the A10), -Dimagination-g9p-es3: what the G9P paths do, as the
 * physical device reports it.
 *
 * Upstream's tables describe Rogue (four colour attachments, 4x MSAA,
 * transform feedback, every depth format). Zink turns them into an OpenGL
 * ES context without checking any of them, so an application that relies on
 * one of those features gets nothing on the screen and no error. With the
 * option on, the G9P device reports what its own paths implement; each
 * feature commit of the series raises its own numbers here.
 */

#include "vk_device.h"
#include "vk_physical_device.h"

#include "pvr_g9p_es3.h"

void
pvr_g9p_es3_limits(struct vk_device_extension_table *ext, struct vk_features *features,
                   struct vk_properties *props)
{
   /*
    * Colour targets: 1, 2 or 4 of the same 8-bit RGBA format, without
    * blending past the first (pvr_arch_g9p_es3_gfx.h); three are refused
    * when a pipeline is created
    */
   props->maxColorAttachments = 4;
   props->maxFragmentOutputAttachments = 4;
   features->independentBlend = false;

   /* Textures: 2D, 2D arrays and cube maps (pvr_arch_g9p_es3_gfx.h); no cube arrays */
   features->imageCubeArray = false;

   /*
    * 1 or 4 samples (pvr_arch_g9p_es3_gfx.h "4x MSAA"); a 4x image is
    * rendered and resolved, never sampled
    */
   props->framebufferColorSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT;
   props->framebufferDepthSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT;
   props->framebufferStencilSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT;
   props->framebufferNoAttachmentsSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT;
   props->standardSampleLocations = true;
   props->sampledImageColorSampleCounts = VK_SAMPLE_COUNT_1_BIT;
   props->sampledImageIntegerSampleCounts = VK_SAMPLE_COUNT_1_BIT;
   props->sampledImageDepthSampleCounts = VK_SAMPLE_COUNT_1_BIT;
   props->sampledImageStencilSampleCounts = VK_SAMPLE_COUNT_1_BIT;
   features->sampleRateShading = false;
   features->alphaToOne = false;

   /*
    * Transform feedback: the vertex shader stores the records of
    * non-indexed list draws, stream 0, up to four buffers
    * (pvr_arch_g9p_es3_gfx.h); no queries, no draws from a counter
    */
   ext->EXT_transform_feedback = true;
   features->transformFeedback = true;
   features->geometryStreams = false;
   props->maxTransformFeedbackStreams = 1;
   props->maxTransformFeedbackBuffers = 4;
   props->transformFeedbackQueries = false;
   props->transformFeedbackStreamsLinesTriangles = false;
   props->transformFeedbackRasterizationStreamSelect = false;
   props->transformFeedbackDraw = false;
}

VkSampleCountFlags
pvr_g9p_es3_sample_counts(VkImageUsageFlags usage)
{
   /* sampling and storing a multisample image are not built */
   if (usage & (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT))
      return VK_SAMPLE_COUNT_1_BIT;
   return VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT;
}

VkFormatFeatureFlags2
pvr_g9p_es3_ds_features(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_D16_UNORM:
   case VK_FORMAT_D32_SFLOAT:
   case VK_FORMAT_S8_UINT:
   case VK_FORMAT_D32_SFLOAT_S8_UINT:
      return VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT;
   default:
      return 0;
   }
}
