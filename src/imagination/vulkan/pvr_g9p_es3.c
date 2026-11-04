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

   /* One sample: the render job streams carry no MSAA words */
   props->framebufferColorSampleCounts = VK_SAMPLE_COUNT_1_BIT;
   props->framebufferDepthSampleCounts = VK_SAMPLE_COUNT_1_BIT;
   props->framebufferStencilSampleCounts = VK_SAMPLE_COUNT_1_BIT;
   props->framebufferNoAttachmentsSampleCounts = VK_SAMPLE_COUNT_1_BIT;
   props->sampledImageColorSampleCounts = VK_SAMPLE_COUNT_1_BIT;
   props->sampledImageIntegerSampleCounts = VK_SAMPLE_COUNT_1_BIT;
   props->sampledImageDepthSampleCounts = VK_SAMPLE_COUNT_1_BIT;
   props->sampledImageStencilSampleCounts = VK_SAMPLE_COUNT_1_BIT;
   features->sampleRateShading = false;
   features->alphaToOne = false;

   /* Transform feedback: the stream-out program is a HALT, nothing is stored */
   ext->EXT_transform_feedback = false;
   features->transformFeedback = false;
   features->geometryStreams = false;
   props->maxTransformFeedbackStreams = 0;
   props->maxTransformFeedbackBuffers = 0;
   props->transformFeedbackQueries = false;
   props->transformFeedbackDraw = false;
}

VkSampleCountFlags
pvr_g9p_es3_sample_counts(void)
{
   return VK_SAMPLE_COUNT_1_BIT;
}
