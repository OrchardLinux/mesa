/* SPDX-License-Identifier: MIT */
/* Smoke and draw test of G5P on pvr_*, on the device (or offline over a fake node). On an A9, the PowerVR driver
 * built with -Dimagination-g5p-experimental=true must find the apple_g5p node and create a device (pvr_* device setup
 * through the apple_g5p winsys and the G5P device hook). Every device runs the winsys's memory paths: allocate, map,
 * write and read back, bind a buffer at an offset, free. With --submit also the queue's sync path: two empty
 * submissions (the winsys's null job: DRM syncobjs, no GPU work) chained by a semaphore, a fence waited.
 * On any other Vulkan device (an A10's G9P path, lavapipe) the same options run unchanged.
 *
 *   triangle [--cycles N] [--bos PATH] [--submit]
 *
 * --cycles N  create and destroy the VkDevice N times in one instance (default 1); every device runs the memory
 *             paths. Each VkDevice has its own winsys (VM and arena); the instance's physical device has one more.
 * --bos PATH  the kernel's debugfs g5p_bos (root only): this process's BO and file lines are counted at each step.
 *             After every device destroy this process must hold what it held before the device (the physical
 *             device's winsys alone), and after the instance destroy nothing: no BO and no open file. Except BOs
 *             the kernel holds for a closed render context (render_user_close_defer: a context the PM ran last
 *             or second to last stays bound until another context's job has moved the PM past it; its VM keeps
 *             its BOs pinned): a BO of this process with no handle, not exported or imported, and VM-pinned can
 *             only be that, since its file is closed. At most 2 (the kernel's close_deferred slots); they are
 *             polled for 5 s, then reported as "held by close-defer", not as a leak.
 * --submit   also the two empty submissions per device. Needs the kernel's SYNCOBJ_TRANSFER (DRIVER_SYNCOBJ_TIMELINE,
 *             as the A9 driver has); the offline fake node only knows binary syncobjs, so it is off by default.
 * --render   also one clear-only render pass per device (G5P-PVR-RENDER line); --verify also compares its pixels.
 * --triangle also one render pass that clears and draws one triangle per device (G5P-PVR-TRIANGLE
 *             line): pass.vert + solid.frag (a constant red, so the test is exact), the vertices and the negative-height
 *             viewport of our reference general-triangle scene. With --verify, every pixel at least 1 px from a
 *             triangle edge must be exact (red inside, the clear colour outside); the edge band is reported apart and
 *             may hold only those two colours.
 * --load     also two render passes on one image per device (G5P-PVR-LOAD line): pass 1 clears and
 *             draws triangle 1 in red (left); pass 2 has loadOp LOAD and draws triangle 2 in green (right). With
 *             --verify, every pixel at least 1 px from an edge must be exact: red in triangle 1 (only a real load
 *             keeps it), green in triangle 2, the clear colour elsewhere; the edge bands are reported apart.
 * Prints "G5P-PVR-SMOKE PASS ..." or "G5P-PVR-SMOKE FAIL <step> ...". */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

static const char *bos_path;
static unsigned closed_held;
static unsigned long long closed_held_kib;
static int do_submit, do_render, do_verify, do_triangle, do_load;
static char tag[32];

struct held {
   unsigned bos, files, deferred;
   unsigned long long kib, pinned_kib, deferred_kib;
};

/* the kernel's close_deferred[] slots (drm-render-unbind.inc) */
#define CLOSE_DEFER_MAX 2

/* This process's lines in g5p_bos: "bo size=<K>K ... by=<comm>[<pid>]" and "file <comm>[<pid>] ...", plus the
 * device total "live=... device_pinned=<K>K". */
static int held(struct held *h)
{
   memset(h, 0, sizeof(*h));
   if (!bos_path)
      return 0;
   FILE *f = fopen(bos_path, "r");
   if (!f) {
      printf("G5P-PVR-SMOKE FAIL cannot read %s (root only)\n", bos_path);
      return -1;
   }
   char line[512];
   while (fgets(line, sizeof(line), f)) {
      const char *p;
      if (!strncmp(line, "bo ", 3) && strstr(line, tag)) {
         const unsigned long long kib = strtoull(line + 8, NULL, 10); /* "bo size=" */
         h->bos++;
         h->kib += kib;
         /* no handle, not shared, pinned by a VM: a closed render context's, held by the kernel */
         if (strstr(line, " handles=0 ") && strstr(line, " exported=n imported=n ") && !strstr(line, " vm_pins=0 ")) {
            h->deferred++;
            h->deferred_kib += kib;
         }
      } else if (!strncmp(line, "file ", 5) && strstr(line, tag)) {
         h->files++;
      } else if (!strncmp(line, "live=", 5) && (p = strstr(line, "device_pinned="))) {
         h->pinned_kib = strtoull(p + 14, NULL, 10);
      }
   }
   fclose(f);
   return 0;
}

static void show(const char *step, const struct held *h)
{
   if (bos_path)
      printf("held %-22s bos=%u (%llu KiB) files=%u device_pinned=%llu KiB%s\n", step, h->bos, h->kib, h->files,
             h->pinned_kib, h->deferred ? " (some held by close-defer)" : "");
}

/* Kernel-held BOs (see --bos): poll up to 5 s for their release (another context's job releases them), then
 * report what is still held. Returns -1 if it cannot be close-defer (more than the kernel's slots). */
static int close_defer(const char *when, struct held *h)
{
   if (!h->deferred)
      return 0;
   for (unsigned i = 0; i < 50 && h->deferred; i++) {
      usleep(100000);
      if (held(h))
         return -1;
   }
   if (h->deferred > CLOSE_DEFER_MAX) {
      printf("G5P-PVR-SMOKE FAIL %s: %u BOs without a handle stay VM-pinned, more than the kernel's %u "
             "close-deferred contexts\n", when, h->deferred, CLOSE_DEFER_MAX);
      return -1;
   }
   closed_held = 0;
   closed_held_kib = 0;
   if (h->deferred) {
      closed_held = h->deferred;
      closed_held_kib = h->deferred_kib;
      printf("held by close-defer %-10s %u BOs (%llu KiB) after 5 s: the kernel keeps a closed render context "
             "until another context's job moves the PM past it\n", when, h->deferred, h->deferred_kib);
   }
   return 0;
}

#define CHECK(x)                                                                                                     \
   do {                                                                                                              \
      VkResult r_ = (x);                                                                                             \
      if (r_ != VK_SUCCESS) {                                                                                        \
         printf("G5P-PVR-SMOKE FAIL cycle %u: %s = %d\n", cycle, #x, r_);                                           \
         return 1;                                                                                                   \
      }                                                                                                              \
   } while (0)

/* The shaders (shaders.glsl), compiled by build.sh with glslangValidator into this generated header:
 * g5p_pass_vert, g5p_solid_frag (--triangle) and g5p_green_frag (--load's second colour). */
#include "triangle-spirv.h"

/* the general-triangle scene's vertices (vec4 positions) and its viewport (0, 64, 64, -64): the triangle covers
 * (8,56) (56,56) (32,8) in pixels */
static const float tri[12] = {-.75f, -.75f, 0, 1, .75f, -.75f, 0, 1, 0, .75f, 0, 1};

/* host-visible memory for a resource */
static int bind_host_memory(VkPhysicalDevice pd, VkDevice dev, const VkMemoryRequirements *req, VkDeviceMemory *mem,
                            unsigned cycle)
{
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   uint32_t type = UINT32_MAX;
   for (uint32_t i = 0; i < mp.memoryTypeCount && type == UINT32_MAX; i++)
      if ((req->memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
         type = i;
   if (type == UINT32_MAX) {
      printf("G5P-PVR-SMOKE FAIL cycle %u: no host-visible memory type\n", cycle);
      return 1;
   }
   VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req->size,
                               .memoryTypeIndex = type};
   CHECK(vkAllocateMemory(dev, &mai, NULL, mem));
   return 0;
}

/* Signed distance of pixel centre (px, py) to the edge a->b, positive on the triangle's inside (orientation o) */
static double edge_dist(double px, double py, const double *a, const double *b, double o)
{
   const double ex = b[0] - a[0], ey = b[1] - a[1];
   return o * (ex * (py - a[1]) - ey * (px - a[0])) / sqrt(ex * ex + ey * ey);
}

/* --render: one render pass that only clears a 64x64 linear B8G8R8A8_UNORM image (no draw): pvr_* records it,
 * the G5P hooks give it its background clear and end of tile, the apple_g5p winsys submits the TYPED_V3 pass.
 * --verify also compares every pixel with the clear colour (on the device; the fake node executes nothing). */
#define RW 64
#define RH 64
static int render_pass(VkPhysicalDevice pd, VkDevice dev, unsigned cycle, int triangle)
{
   VkImageCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                            .format = VK_FORMAT_B8G8R8A8_UNORM, .extent = {RW, RH, 1}, .mipLevels = 1,
                            .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_LINEAR,
                            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
   VkImage img;
   CHECK(vkCreateImage(dev, &ici, NULL, &img));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, img, &req);
   VkDeviceMemory mem;
   if (bind_host_memory(pd, dev, &req, &mem, cycle))
      return 1;
   CHECK(vkBindImageMemory(dev, img, mem, 0));
   VkImageSubresource sub = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT};
   VkSubresourceLayout lay;
   vkGetImageSubresourceLayout(dev, img, &sub, &lay);
   uint8_t *px;
   CHECK(vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, (void **)&px));
   memset(px, 0x5a, req.size);

   VkImageViewCreateInfo vci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = ici.format,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
   VkImageView view;
   CHECK(vkCreateImageView(dev, &vci, NULL, &view));
   VkAttachmentDescription att = {.format = ici.format, .samples = VK_SAMPLE_COUNT_1_BIT,
                                  .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                  .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                  .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                  .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                  .finalLayout = VK_IMAGE_LAYOUT_GENERAL};
   VkAttachmentReference ref = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
   VkSubpassDescription sp = {.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS, .colorAttachmentCount = 1,
                              .pColorAttachments = &ref};
   VkRenderPassCreateInfo rci = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, .attachmentCount = 1,
                                 .pAttachments = &att, .subpassCount = 1, .pSubpasses = &sp};
   VkRenderPass rp;
   CHECK(vkCreateRenderPass(dev, &rci, NULL, &rp));
   VkFramebufferCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = rp,
                                  .attachmentCount = 1, .pAttachments = &view, .width = RW, .height = RH,
                                  .layers = 1};
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(dev, &fci, NULL, &fb));

   /* --triangle: the vertex buffer and the pipeline (pass.vert + solid.frag, dynamic viewport and scissor) */
   VkBuffer vbuf = VK_NULL_HANDLE;
   VkDeviceMemory vmem = VK_NULL_HANDLE;
   VkShaderModule vsm = VK_NULL_HANDLE, fsm = VK_NULL_HANDLE;
   VkPipelineLayout layout = VK_NULL_HANDLE;
   VkPipeline pipe = VK_NULL_HANDLE;
   if (triangle) {
      VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = sizeof(tri),
                                .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT};
      CHECK(vkCreateBuffer(dev, &bci, NULL, &vbuf));
      VkMemoryRequirements vreq;
      vkGetBufferMemoryRequirements(dev, vbuf, &vreq);
      if (bind_host_memory(pd, dev, &vreq, &vmem, cycle))
         return 1;
      CHECK(vkBindBufferMemory(dev, vbuf, vmem, 0));
      void *vp;
      CHECK(vkMapMemory(dev, vmem, 0, VK_WHOLE_SIZE, 0, &vp));
      memcpy(vp, tri, sizeof(tri));
      vkUnmapMemory(dev, vmem);

      VkShaderModuleCreateInfo smi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                      .codeSize = sizeof(g5p_pass_vert), .pCode = g5p_pass_vert};
      CHECK(vkCreateShaderModule(dev, &smi, NULL, &vsm));
      smi.codeSize = sizeof(g5p_solid_frag);
      smi.pCode = g5p_solid_frag;
      CHECK(vkCreateShaderModule(dev, &smi, NULL, &fsm));
      VkPipelineLayoutCreateInfo lci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      CHECK(vkCreatePipelineLayout(dev, &lci, NULL, &layout));
      VkPipelineShaderStageCreateInfo stages[2] = {
         {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
          .module = vsm, .pName = "main"},
         {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
          .module = fsm, .pName = "main"}};
      VkVertexInputBindingDescription vib = {.binding = 0, .stride = 16, .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};
      VkVertexInputAttributeDescription via = {.location = 0, .binding = 0,
                                               .format = VK_FORMAT_R32G32B32A32_SFLOAT, .offset = 0};
      VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                                                 .vertexBindingDescriptionCount = 1,
                                                 .pVertexBindingDescriptions = &vib,
                                                 .vertexAttributeDescriptionCount = 1,
                                                 .pVertexAttributeDescriptions = &via};
      VkPipelineInputAssemblyStateCreateInfo ia = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
         .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
      VkPipelineViewportStateCreateInfo vps = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                               .viewportCount = 1, .scissorCount = 1};
      VkPipelineRasterizationStateCreateInfo rs = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .polygonMode = VK_POLYGON_MODE_FILL,
         .cullMode = VK_CULL_MODE_NONE, .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f};
      VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                                 .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
      VkPipelineColorBlendAttachmentState cba = {.colorWriteMask = 0xf};
      VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                                .attachmentCount = 1, .pAttachments = &cba};
      const VkDynamicState dyns[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
      VkPipelineDynamicStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                             .dynamicStateCount = 2, .pDynamicStates = dyns};
      VkGraphicsPipelineCreateInfo gci = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = 2,
                                          .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
                                          .pViewportState = &vps, .pRasterizationState = &rs,
                                          .pMultisampleState = &ms, .pColorBlendState = &cb,
                                          .pDynamicState = &ds, .layout = layout, .renderPass = rp};
      CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gci, NULL, &pipe));
   }

   VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(dev, &pci, NULL, &pool));
   VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(dev, &cai, &cb));
   VkCommandBufferBeginInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CHECK(vkBeginCommandBuffer(cb, &cbi));
   /* B8G8R8A8: blue 51, green 102, red 153, alpha 204 in memory order */
   VkClearValue cv = {.color.float32 = {153 / 255.0f, 102 / 255.0f, 51 / 255.0f, 204 / 255.0f}};
   VkRenderPassBeginInfo rbi = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = rp,
                                .framebuffer = fb, .renderArea = {{0, 0}, {RW, RH}}, .clearValueCount = 1,
                                .pClearValues = &cv};
   vkCmdBeginRenderPass(cb, &rbi, VK_SUBPASS_CONTENTS_INLINE);
   if (triangle) {
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
      const VkDeviceSize zero = 0;
      vkCmdBindVertexBuffers(cb, 0, 1, &vbuf, &zero);
      const VkViewport vpt = {0, RH, RW, -(float)RH, 0, 1};
      const VkRect2D sc = {{0, 0}, {RW, RH}};
      vkCmdSetViewport(cb, 0, 1, &vpt);
      vkCmdSetScissor(cb, 0, 1, &sc);
      vkCmdDraw(cb, 3, 1, 0, 0);
   }
   vkCmdEndRenderPass(cb);
   CHECK(vkEndCommandBuffer(cb));
   VkQueue q;
   vkGetDeviceQueue(dev, 0, 0, &q);
   VkFenceCreateInfo fnci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   CHECK(vkCreateFence(dev, &fnci, NULL, &fence));
   VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
   CHECK(vkQueueSubmit(q, 1, &si, fence));
   CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 5000000000ull));

   /* the clear colour; the triangle's red (B8G8R8A8) */
   const uint8_t clear_px[4] = {51, 102, 153, 204}, red_px[4] = {0, 0, 255, 255};
   /* the triangle in pixels and its orientation */
   const double v[3][2] = {{8, 56}, {56, 56}, {32, 8}};
   const double o = ((v[1][0] - v[0][0]) * (v[2][1] - v[0][1]) - (v[1][1] - v[0][1]) * (v[2][0] - v[0][0])) > 0 ? 1 : -1;
   unsigned bad = 0, fx = 0, fy = 0, inside = 0, outside = 0, band = 0, band_red = 0, band_clear = 0;
   if (do_verify) {
      for (unsigned y = 0; y < RH; y++)
         for (unsigned x = 0; x < RW; x++) {
            const uint8_t *p = px + lay.offset + y * lay.rowPitch + 4 * x;
            const uint8_t *want = clear_px;
            if (triangle) {
               double d = 1e9;
               for (unsigned e = 0; e < 3; e++)
                  d = fmin(d, edge_dist(x + 0.5, y + 0.5, v[e], v[(e + 1) % 3], o));
               if (d < 1.0 && d > -1.0) {
                  /* the edge band: either colour, counted apart */
                  band++;
                  if (!memcmp(p, red_px, 4))
                     band_red++;
                  else if (!memcmp(p, clear_px, 4))
                     band_clear++;
                  else if (!bad++)
                     fx = x, fy = y;
                  continue;
               }
               want = d >= 1.0 ? red_px : clear_px;
               if (d >= 1.0)
                  inside++;
               else
                  outside++;
            }
            if (memcmp(p, want, 4)) {
               if (!bad)
                  fx = x, fy = y;
               bad++;
            }
         }
   }
   const uint8_t *f = px + lay.offset + fy * lay.rowPitch + 4 * fx;
   if (triangle) {
      printf("G5P-PVR-TRIANGLE %s cycle %u: clear and one triangle %ux%u, row pitch %llu, ", bad ? "FAIL" : "PASS",
             cycle, RW, RH, (unsigned long long)lay.rowPitch);
      if (do_verify)
         printf("%u inside and %u outside %s, edge band %u (%u red, %u clear, %u other)", inside, outside,
                bad ? "compared" : "exact", band, band_red, band_clear, band - band_red - band_clear);
      else
         printf("submitted and retired (not compared)");
   } else {
      printf("G5P-PVR-RENDER %s cycle %u: clear-only pass %ux%u, row pitch %llu, %s", bad ? "FAIL" : "PASS", cycle,
             RW, RH, (unsigned long long)lay.rowPitch,
             do_verify ? "pixels compared" : "submitted and retired (not compared)");
   }
   if (bad)
      printf(", %u pixels differ, first at %u,%u: %u %u %u %u", bad, fx, fy, f[0], f[1], f[2], f[3]);
   printf("\n");

   vkDestroyFence(dev, fence, NULL);
   if (triangle) {
      vkDestroyPipeline(dev, pipe, NULL);
      vkDestroyPipelineLayout(dev, layout, NULL);
      vkDestroyShaderModule(dev, vsm, NULL);
      vkDestroyShaderModule(dev, fsm, NULL);
      vkDestroyBuffer(dev, vbuf, NULL);
      vkFreeMemory(dev, vmem, NULL);
   }
   vkFreeCommandBuffers(dev, pool, 1, &cb);
   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyRenderPass(dev, rp, NULL);
   vkDestroyImageView(dev, view, NULL);
   vkUnmapMemory(dev, mem);
   vkDestroyImage(dev, img, NULL);
   vkFreeMemory(dev, mem, NULL);
   return bad ? 1 : 0;
}

/* --load: a pipeline drawing a flat colour (pass.vert + fs), dynamic viewport and scissor */
static int load_pipeline(VkDevice dev, VkRenderPass rp, const uint32_t *fs, size_t fs_size, VkPipelineLayout layout,
                         VkPipeline *pipe, unsigned cycle)
{
   VkShaderModule vsm, fsm;
   VkShaderModuleCreateInfo smi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                   .codeSize = sizeof(g5p_pass_vert), .pCode = g5p_pass_vert};
   CHECK(vkCreateShaderModule(dev, &smi, NULL, &vsm));
   smi.codeSize = fs_size;
   smi.pCode = fs;
   CHECK(vkCreateShaderModule(dev, &smi, NULL, &fsm));
   VkPipelineShaderStageCreateInfo stages[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
       .module = vsm, .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
       .module = fsm, .pName = "main"}};
   VkVertexInputBindingDescription vib = {.binding = 0, .stride = 16, .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};
   VkVertexInputAttributeDescription via = {.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32A32_SFLOAT};
   VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                                              .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &vib,
                                              .vertexAttributeDescriptionCount = 1,
                                              .pVertexAttributeDescriptions = &via};
   VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   VkPipelineViewportStateCreateInfo vps = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                            .viewportCount = 1, .scissorCount = 1};
   VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
                                                .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f};
   VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                              .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
   VkPipelineColorBlendAttachmentState cba = {.colorWriteMask = 0xf};
   VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                             .attachmentCount = 1, .pAttachments = &cba};
   const VkDynamicState dyns[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
   VkPipelineDynamicStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                          .dynamicStateCount = 2, .pDynamicStates = dyns};
   VkGraphicsPipelineCreateInfo gci = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = 2,
                                       .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
                                       .pViewportState = &vps, .pRasterizationState = &rs, .pMultisampleState = &ms,
                                       .pColorBlendState = &cb, .pDynamicState = &ds, .layout = layout,
                                       .renderPass = rp};
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gci, NULL, pipe));
   vkDestroyShaderModule(dev, vsm, NULL);
   vkDestroyShaderModule(dev, fsm, NULL);
   return 0;
}

/* --load: pass 1 (CLEAR) draws triangle 1, pass 2 (LOAD) triangle 2; each pass its own submit and fence */
static const float load_tris[24] = {-.9f, -.75f, 0, 1, -.1f, -.75f, 0, 1, -.5f, .75f, 0, 1,
                                    .1f, -.75f, 0, 1, .9f, -.75f, 0, 1, .5f, .75f, 0, 1};
static int render_load(VkPhysicalDevice pd, VkDevice dev, unsigned cycle)
{
   VkImageCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                            .format = VK_FORMAT_B8G8R8A8_UNORM, .extent = {RW, RH, 1}, .mipLevels = 1,
                            .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_LINEAR,
                            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
   VkImage img;
   CHECK(vkCreateImage(dev, &ici, NULL, &img));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, img, &req);
   VkDeviceMemory mem;
   if (bind_host_memory(pd, dev, &req, &mem, cycle))
      return 1;
   CHECK(vkBindImageMemory(dev, img, mem, 0));
   VkImageSubresource sub = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT};
   VkSubresourceLayout lay;
   vkGetImageSubresourceLayout(dev, img, &sub, &lay);
   uint8_t *px;
   CHECK(vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, (void **)&px));
   memset(px, 0x5a, req.size);
   VkImageViewCreateInfo vci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = ici.format,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
   VkImageView view;
   CHECK(vkCreateImageView(dev, &vci, NULL, &view));

   /* pass 1 clears, pass 2 loads; both store, the image stays GENERAL between them */
   VkRenderPass rp[2];
   for (unsigned i = 0; i < 2; i++) {
      VkAttachmentDescription att = {.format = ici.format, .samples = VK_SAMPLE_COUNT_1_BIT,
                                     .loadOp = i ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR,
                                     .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                     .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                     .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                     .initialLayout = i ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
                                     .finalLayout = VK_IMAGE_LAYOUT_GENERAL};
      VkAttachmentReference ref = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
      VkSubpassDescription sp = {.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS, .colorAttachmentCount = 1,
                                 .pColorAttachments = &ref};
      VkRenderPassCreateInfo rci = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, .attachmentCount = 1,
                                    .pAttachments = &att, .subpassCount = 1, .pSubpasses = &sp};
      CHECK(vkCreateRenderPass(dev, &rci, NULL, &rp[i]));
   }
   VkFramebufferCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = rp[0],
                                  .attachmentCount = 1, .pAttachments = &view, .width = RW, .height = RH,
                                  .layers = 1};
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(dev, &fci, NULL, &fb));

   VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = sizeof(load_tris),
                             .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT};
   VkBuffer vbuf;
   CHECK(vkCreateBuffer(dev, &bci, NULL, &vbuf));
   VkMemoryRequirements vreq;
   vkGetBufferMemoryRequirements(dev, vbuf, &vreq);
   VkDeviceMemory vmem;
   if (bind_host_memory(pd, dev, &vreq, &vmem, cycle))
      return 1;
   CHECK(vkBindBufferMemory(dev, vbuf, vmem, 0));
   void *vp;
   CHECK(vkMapMemory(dev, vmem, 0, VK_WHOLE_SIZE, 0, &vp));
   memcpy(vp, load_tris, sizeof(load_tris));
   vkUnmapMemory(dev, vmem);

   VkPipelineLayoutCreateInfo lci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(dev, &lci, NULL, &layout));
   VkPipeline pipe[2];
   if (load_pipeline(dev, rp[0], g5p_solid_frag, sizeof(g5p_solid_frag), layout, &pipe[0], cycle) ||
       load_pipeline(dev, rp[1], g5p_green_frag, sizeof(g5p_green_frag), layout, &pipe[1], cycle))
      return 1;

   VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(dev, &pci, NULL, &pool));
   VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 2};
   VkCommandBuffer cb[2];
   CHECK(vkAllocateCommandBuffers(dev, &cai, cb));
   VkQueue q;
   vkGetDeviceQueue(dev, 0, 0, &q);
   VkFenceCreateInfo fnci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   CHECK(vkCreateFence(dev, &fnci, NULL, &fence));
   /* the clear colour of step 6 */
   VkClearValue cv = {.color.float32 = {153 / 255.0f, 102 / 255.0f, 51 / 255.0f, 204 / 255.0f}};
   for (unsigned i = 0; i < 2; i++) {
      VkCommandBufferBeginInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      CHECK(vkBeginCommandBuffer(cb[i], &cbi));
      VkRenderPassBeginInfo rbi = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = rp[i],
                                   .framebuffer = fb, .renderArea = {{0, 0}, {RW, RH}},
                                   .clearValueCount = i ? 0 : 1, .pClearValues = i ? NULL : &cv};
      vkCmdBeginRenderPass(cb[i], &rbi, VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(cb[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipe[i]);
      const VkDeviceSize zero = 0;
      vkCmdBindVertexBuffers(cb[i], 0, 1, &vbuf, &zero);
      const VkViewport vpt = {0, RH, RW, -(float)RH, 0, 1};
      const VkRect2D sc = {{0, 0}, {RW, RH}};
      vkCmdSetViewport(cb[i], 0, 1, &vpt);
      vkCmdSetScissor(cb[i], 0, 1, &sc);
      vkCmdDraw(cb[i], 3, 1, 3 * i, 0);
      vkCmdEndRenderPass(cb[i]);
      CHECK(vkEndCommandBuffer(cb[i]));
      VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb[i]};
      CHECK(vkResetFences(dev, 1, &fence));
      CHECK(vkQueueSubmit(q, 1, &si, fence));
      CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 5000000000ull));
   }

   /* B8G8R8A8: the clear colour, red (triangle 1), green (triangle 2); the triangles in pixels */
   const uint8_t clear_px[4] = {51, 102, 153, 204}, col[2][4] = {{0, 0, 255, 255}, {0, 255, 0, 255}};
   double v[2][3][2];
   for (unsigned t = 0; t < 2; t++)
      for (unsigned k = 0; k < 3; k++) {
         v[t][k][0] = RW / 2.0 + RW / 2.0 * load_tris[12 * t + 4 * k];
         v[t][k][1] = RH / 2.0 - RH / 2.0 * load_tris[12 * t + 4 * k + 1];
      }
   unsigned bad = 0, fx = 0, fy = 0, inside[2] = {0}, outside = 0, band = 0, band_ok = 0;
   if (do_verify) {
      for (unsigned y = 0; y < RH; y++)
         for (unsigned x = 0; x < RW; x++) {
            const uint8_t *p = px + lay.offset + y * lay.rowPitch + 4 * x;
            const uint8_t *want = clear_px;
            int in_band = 0;
            for (unsigned t = 0; t < 2; t++) {
               const double o = ((v[t][1][0] - v[t][0][0]) * (v[t][2][1] - v[t][0][1]) -
                                 (v[t][1][1] - v[t][0][1]) * (v[t][2][0] - v[t][0][0])) > 0 ? 1 : -1;
               double d = 1e9;
               for (unsigned e = 0; e < 3; e++)
                  d = fmin(d, edge_dist(x + 0.5, y + 0.5, v[t][e], v[t][(e + 1) % 3], o));
               if (d < 1.0 && d > -1.0) {
                  in_band = 1;
                  if (!memcmp(p, col[t], 4) || !memcmp(p, clear_px, 4))
                     band_ok++;
                  else if (!bad++)
                     fx = x, fy = y;
                  break;
               }
               if (d >= 1.0) {
                  want = col[t];
                  inside[t]++;
               }
            }
            if (in_band) {
               band++;
               continue;
            }
            if (want == clear_px)
               outside++;
            if (memcmp(p, want, 4)) {
               if (!bad)
                  fx = x, fy = y;
               bad++;
            }
         }
   }
   const uint8_t *f = px + lay.offset + fy * lay.rowPitch + 4 * fx;
   printf("G5P-PVR-LOAD %s cycle %u: pass 1 clear + triangle 1, pass 2 LOAD + triangle 2, %ux%u, row pitch %llu, ",
          bad ? "FAIL" : "PASS", cycle, RW, RH, (unsigned long long)lay.rowPitch);
   if (do_verify)
      printf("%u in triangle 1 (red), %u in triangle 2 (green), %u outside %s, edge bands %u (%u other)", inside[0],
             inside[1], outside, bad ? "compared" : "exact", band, band - band_ok);
   else
      printf("submitted and retired (not compared)");
   if (bad)
      printf(", %u pixels differ, first at %u,%u: %u %u %u %u", bad, fx, fy, f[0], f[1], f[2], f[3]);
   printf("\n");
   /* The bytes (B G R A) at three points: the clear area, triangle 1's and triangle 2's centroids */
   {
      unsigned pt[3][2] = {{2, 2}, {0, 0}, {0, 0}};
      for (unsigned t = 0; t < 2; t++) {
         pt[1 + t][0] = (unsigned)((v[t][0][0] + v[t][1][0] + v[t][2][0]) / 3);
         pt[1 + t][1] = (unsigned)((v[t][0][1] + v[t][1][1] + v[t][2][1]) / 3);
      }
      const char *name[3] = {"clear", "tri1", "tri2"};
      printf("G5P-PVR-LOAD-PIXELS");
      for (unsigned i = 0; i < 3; i++) {
         const uint8_t *q = px + lay.offset + pt[i][1] * lay.rowPitch + 4 * pt[i][0];
         printf(" %s(%u,%u)=%u %u %u %u", name[i], pt[i][0], pt[i][1], q[0], q[1], q[2], q[3]);
      }
      printf("\n");
   }

   vkDestroyFence(dev, fence, NULL);
   vkFreeCommandBuffers(dev, pool, 2, cb);
   vkDestroyCommandPool(dev, pool, NULL);
   for (unsigned i = 0; i < 2; i++) {
      vkDestroyPipeline(dev, pipe[i], NULL);
      vkDestroyRenderPass(dev, rp[i], NULL);
   }
   vkDestroyPipelineLayout(dev, layout, NULL);
   vkDestroyBuffer(dev, vbuf, NULL);
   vkFreeMemory(dev, vmem, NULL);
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyImageView(dev, view, NULL);
   vkUnmapMemory(dev, mem);
   vkDestroyImage(dev, img, NULL);
   vkFreeMemory(dev, mem, NULL);
   return bad ? 1 : 0;
}

/* One VkDevice: 1 MiB of memory mapped, written and read back, a buffer bound at an offset */
static int device_cycle(VkPhysicalDevice pd, unsigned cycle, VkDeviceSize *bound_at)
{
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1,
                                  .pQueuePriorities = &prio};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qci};
   VkDevice dev;
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));

   const VkDeviceSize size = 1 << 20;
   VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = size};
   VkDeviceMemory mem;
   CHECK(vkAllocateMemory(dev, &mai, NULL, &mem));
   unsigned char *p;
   CHECK(vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, (void **)&p));
   for (VkDeviceSize i = 0; i < size; i++)
      p[i] = (unsigned char)(i * 7 + 1 + cycle);
   for (VkDeviceSize i = 0; i < size; i++)
      if (p[i] != (unsigned char)(i * 7 + 1 + cycle)) {
         printf("G5P-PVR-SMOKE FAIL cycle %u: readback at %llu\n", cycle, (unsigned long long)i);
         return 1;
      }

   /* a buffer bound at an offset: the winsys returns the slice's address + offset */
   VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4096,
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
   VkBuffer buf;
   CHECK(vkCreateBuffer(dev, &bci, NULL, &buf));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, buf, &req);
   const VkDeviceSize off = (65536 + req.alignment - 1) / req.alignment * req.alignment;
   CHECK(vkBindBufferMemory(dev, buf, mem, off));
   *bound_at = off;

   /* --submit: two empty submissions (no command buffer: the winsys's null job, DRM syncobjs only, no GPU work):
    * the first signals a semaphore, the second waits for it and signals a fence. */
   if (do_submit) {
   VkQueue q;
   vkGetDeviceQueue(dev, 0, 0, &q);
   VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
   VkSemaphore sem;
   CHECK(vkCreateSemaphore(dev, &sci, NULL, &sem));
   VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   CHECK(vkCreateFence(dev, &fci, NULL, &fence));
   const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
   VkSubmitInfo s1 = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .signalSemaphoreCount = 1, .pSignalSemaphores = &sem};
   VkSubmitInfo s2 = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .waitSemaphoreCount = 1, .pWaitSemaphores = &sem,
                      .pWaitDstStageMask = &stage};
   CHECK(vkQueueSubmit(q, 1, &s1, VK_NULL_HANDLE));
   CHECK(vkQueueSubmit(q, 1, &s2, fence));
   CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 2000000000ull));
   CHECK(vkQueueWaitIdle(q));
   vkDestroyFence(dev, fence, NULL);
   vkDestroySemaphore(dev, sem, NULL);
   }

   if (do_render && render_pass(pd, dev, cycle, 0))
      return 1;
   if (do_triangle && render_pass(pd, dev, cycle, 1))
      return 1;
   if (do_load && render_load(pd, dev, cycle))
      return 1;

   vkDestroyBuffer(dev, buf, NULL);
   vkUnmapMemory(dev, mem);
   vkFreeMemory(dev, mem, NULL);
   vkDestroyDevice(dev, NULL);
   return 0;
}

int main(int argc, char **argv)
{
   unsigned cycles = 1, cycle = 0;
   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--cycles") && i + 1 < argc)
         cycles = (unsigned)strtoul(argv[++i], NULL, 10);
      else if (!strcmp(argv[i], "--bos") && i + 1 < argc)
         bos_path = argv[++i];
      else if (!strcmp(argv[i], "--submit"))
         do_submit = 1;
      else if (!strcmp(argv[i], "--render"))
         do_render = 1;
      else if (!strcmp(argv[i], "--verify"))
         do_verify = 1;
      else if (!strcmp(argv[i], "--triangle"))
         do_triangle = 1;
      else if (!strcmp(argv[i], "--load"))
         do_load = 1;
      else {
         fprintf(stderr, "usage: %s [--cycles N] [--bos PATH] [--submit] [--render] [--triangle] [--load] [--verify]\n",
                 argv[0]);
         return 2;
      }
   }
   /* --verify alone: the clear-only pass (step 6's device rung) */
   if (do_verify && !do_triangle && !do_load)
      do_render = 1;
   if (!cycles)
      cycles = 1;
   snprintf(tag, sizeof(tag), "[%d]", (int)getpid());

   struct held h0, h1, h;
   if (held(&h0))
      return 1;
   show("start", &h0);
   if (h0.bos || h0.files) {
      printf("G5P-PVR-SMOKE FAIL start: this process already holds %u BOs and %u files\n", h0.bos, h0.files);
      return 1;
   }

   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_2};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));

   uint32_t n = 0;
   CHECK(vkEnumeratePhysicalDevices(inst, &n, NULL));
   if (!n) {
      printf("G5P-PVR-SMOKE FAIL no physical device\n");
      return 1;
   }
   VkPhysicalDevice pd[4];
   n = n > 4 ? 4 : n;
   CHECK(vkEnumeratePhysicalDevices(inst, &n, pd));
   VkPhysicalDeviceDriverProperties drv = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
   VkPhysicalDeviceProperties2 props = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &drv};
   vkGetPhysicalDeviceProperties2(pd[0], &props);
   const VkPhysicalDeviceProperties *pp = &props.properties;
   printf("device: %s (vendor %#x device %#x) driver %s [%s] id %d version %#x api %u.%u.%u\n", pp->deviceName,
          pp->vendorID, pp->deviceID, drv.driverName, drv.driverInfo, drv.driverID, pp->driverVersion,
          VK_API_VERSION_MAJOR(pp->apiVersion), VK_API_VERSION_MINOR(pp->apiVersion),
          VK_API_VERSION_PATCH(pp->apiVersion));

   if (held(&h1))
      return 1;
   show("instance", &h1);

   VkDeviceSize off = 0;
   for (cycle = 1; cycle <= cycles; cycle++) {
      if (device_cycle(pd[0], cycle, &off))
         return 1;
      if (held(&h))
         return 1;
      if (cycle == 1 || cycle == cycles)
         show(cycle == 1 ? "after device 1" : "after last device", &h);
      if (h.deferred && close_defer("device", &h))
         return 1;
      if (h.bos - h.deferred != h1.bos || h.kib - h.deferred_kib != h1.kib || h.files != h1.files) {
         show("LEAK", &h);
         printf("G5P-PVR-SMOKE FAIL cycle %u: after the device destroy this process holds %u BOs (%llu KiB) and "
                "%u files, %u (%llu KiB) and %u before the device\n",
                cycle, h.bos, h.kib, h.files, h1.bos, h1.kib, h1.files);
         return 1;
      }
   }

   vkDestroyInstance(inst, NULL);
   if (held(&h))
      return 1;
   show("end", &h);
   if (h.deferred && close_defer("end", &h))
      return 1;
   if (h.bos - h.deferred || h.files) {
      printf("G5P-PVR-SMOKE FAIL end: after the instance destroy this process still holds %u BOs (%llu KiB) and "
             "%u files\n", h.bos, h.kib, h.files);
      return 1;
   }
   char held_note[96] = "";
   if (closed_held)
      snprintf(held_note, sizeof(held_note), "; held by close-defer: %u BOs (%llu KiB)", closed_held,
               closed_held_kib);
   printf("G5P-PVR-SMOKE PASS device %s, %u device cycles, 1 MiB mapped and read back, buffer bound at +%llu%s%s%s\n",
          pp->deviceName, cycles, (unsigned long long)off,
          do_submit ? ", 2 empty submits per device (semaphore, fence) signalled" : "",
          bos_path ? ", BOs and files back to zero" : "", held_note);
   return 0;
}
