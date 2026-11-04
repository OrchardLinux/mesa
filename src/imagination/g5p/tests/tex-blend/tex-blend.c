/* SPDX-License-Identifier: MIT */
/*
 * tex-blend: a hardware test of sampled textures, samplers and blending,
 * for any Vulkan driver (written for the PowerVR driver's Apple backends:
 * G5P on the A9, G9P on the A10). Plain Vulkan 1.0 and no driver internals.
 *
 * Every rung renders into a 64x64 linear B8G8R8A8_UNORM target cleared to
 * CLEAR, reads it back and compares every pixel with a CPU reference:
 *   tex-nearest   a 4x4 RGBA8 texture over the whole target (nearest, clamp)
 *   tex-npot      a 3x5 texture, the same
 *   tex-swizzle   the 4x4 texture through a view swizzled (B, G, R, A)
 *   tex-premult   a 4x4 premultiplied-alpha texture blended ONE,
 *                 ONE_MINUS_SRC_ALPHA (a compositor's window draw)
 *   blend-*       the constant colour SRC over the left half of the target
 *                 with one blend setting each; the right half stays CLEAR
 * Textures are linear and written by the host (no transfer path needed).
 * Sampling is exact. Blended pixels may differ from the CPU's rounding by
 * 1 per channel (the blend arithmetic's precision is the hardware's).
 *
 * usage: tex-blend [--list] [--device N] [--no-verify] [RUNG...]
 * Output: one "TEX-BLEND <rung> PASS|FAIL ..." line per rung, then
 * "TEX-BLEND RESULT PASS|FAIL <n> of <m>". Exit status 0 on PASS.
 * --no-verify runs and retires every rung without comparing (a fake node).
 * Build: ./build.sh [OUT_DIR] (glslangValidator, cc, libvulkan).
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "tex-blend-spirv.h"

#define W 64
#define H 64
#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

/* The clear colour and SOLID_FRAG's colour (shaders.glsl), RGBA */
static const float CLEAR[4] = { 0.2f, 0.4f, 0.6f, 0.8f };
static const float SRC[4] = { 0.5f, 0.3f, 0.1f, 0.6f };
/* blend-constant's dynamic blend constants */
static const float CONSTANTS[4] = { 0.25f, 0.5f, 0.75f, 0.4f };

enum tex_kind { TEX_NONE, TEX_OPAQUE, TEX_PREMULT };

struct rung {
   const char *name;
   enum tex_kind tex;
   unsigned tw, th;   /* texture size */
   bool swizzle;      /* view swizzle (B, G, R, A) */
   bool blend;
   VkBlendFactor src_rgb, dst_rgb, src_a, dst_a;
   VkBlendOp op_rgb, op_a;
   bool constants;    /* dynamic blend constants CONSTANTS */
};

#define F(x) VK_BLEND_FACTOR_##x
#define OP(x) VK_BLEND_OP_##x
static const struct rung rungs[] = {
   { "tex-nearest", TEX_OPAQUE, 4, 4 },
   { "tex-npot", TEX_OPAQUE, 3, 5 },
   { "tex-swizzle", TEX_OPAQUE, 4, 4, true },
   { "tex-premult", TEX_PREMULT, 4, 4, false, true, F(ONE), F(ONE_MINUS_SRC_ALPHA), F(ONE),
     F(ONE_MINUS_SRC_ALPHA), OP(ADD), OP(ADD) },
   { "blend-premult", TEX_NONE, 0, 0, false, true, F(ONE), F(ONE_MINUS_SRC_ALPHA), F(ONE),
     F(ONE_MINUS_SRC_ALPHA), OP(ADD), OP(ADD) },
   { "blend-straight-over", TEX_NONE, 0, 0, false, true, F(SRC_ALPHA), F(ONE_MINUS_SRC_ALPHA), F(ONE),
     F(ONE_MINUS_SRC_ALPHA), OP(ADD), OP(ADD) },
   { "blend-straight", TEX_NONE, 0, 0, false, true, F(SRC_ALPHA), F(ONE_MINUS_SRC_ALPHA), F(SRC_ALPHA),
     F(ONE_MINUS_SRC_ALPHA), OP(ADD), OP(ADD) },
   { "blend-dst-color", TEX_NONE, 0, 0, false, true, F(DST_COLOR), F(ZERO), F(DST_COLOR), F(ZERO), OP(ADD),
     OP(ADD) },
   { "blend-subtract", TEX_NONE, 0, 0, false, true, F(SRC_ALPHA), F(ONE_MINUS_SRC_ALPHA), F(ONE), F(ONE),
     OP(SUBTRACT), OP(ADD) },
   { "blend-constant", TEX_NONE, 0, 0, false, true, F(CONSTANT_COLOR), F(ONE_MINUS_SRC_COLOR),
     F(CONSTANT_COLOR), F(ONE_MINUS_SRC_COLOR), OP(ADD), OP(ADD), true },
   /* the benchmarks' settings: glmark2 pulsar/terrain, glmark2 desktop, vkmark desktop */
   { "blend-sa-one", TEX_NONE, 0, 0, false, true, F(SRC_ALPHA), F(ONE), F(SRC_ALPHA), F(ONE), OP(ADD),
     OP(ADD) },
   { "blend-sa-omsa-z1", TEX_NONE, 0, 0, false, true, F(SRC_ALPHA), F(ONE_MINUS_SRC_ALPHA), F(ZERO), F(ONE),
     OP(ADD), OP(ADD) },
   { "blend-sa-omsa-10", TEX_NONE, 0, 0, false, true, F(SRC_ALPHA), F(ONE_MINUS_SRC_ALPHA), F(ONE), F(ZERO),
     OP(ADD), OP(ADD) },
};

static VkPhysicalDevice pd;
static VkDevice dev;
static VkQueue queue;
static uint32_t queue_family;
static bool verify = true;

#define CHECK(x)                                                                                        \
   do {                                                                                                 \
      VkResult r_ = (x);                                                                                \
      if (r_ != VK_SUCCESS) {                                                                           \
         printf("TEX-BLEND %s FAIL: %s = %d (line %d)\n", cur, #x, r_, __LINE__);                       \
         return 1;                                                                                      \
      }                                                                                                 \
   } while (0)

static const char *cur = "setup";

static uint8_t
unorm8(double v)
{
   v = v < 0 ? 0 : v > 1 ? 1 : v;
   return (uint8_t)lround(v * 255.0);
}

/* A texel of the rung's texture, RGBA */
static void
texel(const struct rung *r, unsigned i, unsigned j, uint8_t out[4])
{
   if (r->tex == TEX_PREMULT) {
      const unsigned a = 100 + 25 * i + 20 * j;

      out[0] = a * i / 4;
      out[1] = a * j / 4;
      out[2] = a / 2;
      out[3] = a;
   } else {
      out[0] = 20 + 50 * i;
      out[1] = 30 + 40 * j;
      out[2] = (i * 7 + j * 13) * 9 % 256;
      out[3] = 255 - 16 * (i + j);
   }
}

static double
factor(VkBlendFactor f, unsigned c, const double *s, const double *d, const double *k)
{
   switch (f) {
   case VK_BLEND_FACTOR_ZERO: return 0;
   case VK_BLEND_FACTOR_ONE: return 1;
   case VK_BLEND_FACTOR_SRC_COLOR: return s[c];
   case VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR: return 1 - s[c];
   case VK_BLEND_FACTOR_DST_COLOR: return d[c];
   case VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR: return 1 - d[c];
   case VK_BLEND_FACTOR_SRC_ALPHA: return s[3];
   case VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA: return 1 - s[3];
   case VK_BLEND_FACTOR_DST_ALPHA: return d[3];
   case VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA: return 1 - d[3];
   case VK_BLEND_FACTOR_CONSTANT_COLOR: return k[c];
   case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR: return 1 - k[c];
   case VK_BLEND_FACTOR_CONSTANT_ALPHA: return k[3];
   case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA: return 1 - k[3];
   case VK_BLEND_FACTOR_SRC_ALPHA_SATURATE: return c < 3 ? fmin(s[3], 1 - d[3]) : 1;
   default: return NAN;
   }
}

/* The CPU's result of blending src (floats) over dst (UNORM8 RGBA), RGBA */
static void
blend_ref(const struct rung *r, const double *s, const uint8_t *dst, uint8_t out[4])
{
   double d[4], k[4];

   for (unsigned c = 0; c < 4; c++) {
      d[c] = dst[c] / 255.0;
      k[c] = CONSTANTS[c];
   }
   for (unsigned c = 0; c < 4; c++) {
      const bool alpha = c == 3;
      const double fs = factor(alpha ? r->src_a : r->src_rgb, c, s, d, k);
      const double fd = factor(alpha ? r->dst_a : r->dst_rgb, c, s, d, k);
      double v;

      switch (alpha ? r->op_a : r->op_rgb) {
      case VK_BLEND_OP_ADD: v = s[c] * fs + d[c] * fd; break;
      case VK_BLEND_OP_SUBTRACT: v = s[c] * fs - d[c] * fd; break;
      case VK_BLEND_OP_REVERSE_SUBTRACT: v = d[c] * fd - s[c] * fs; break;
      case VK_BLEND_OP_MIN: v = fmin(s[c], d[c]); break;
      default: v = fmax(s[c], d[c]); break;
      }
      out[c] = unorm8(v);
   }
}

static int
host_memory(const VkMemoryRequirements *req, VkDeviceMemory *mem)
{
   VkPhysicalDeviceMemoryProperties mp;
   uint32_t type = UINT32_MAX;

   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount && type == UINT32_MAX; i++) {
      if ((req->memoryTypeBits & (1u << i)) &&
          (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
         type = i;
   }
   if (type == UINT32_MAX) {
      printf("TEX-BLEND %s FAIL: no host-visible memory type\n", cur);
      return 1;
   }
   const VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                      .allocationSize = req->size, .memoryTypeIndex = type };
   CHECK(vkAllocateMemory(dev, &mai, NULL, mem));
   return 0;
}

/* A linear 2D image in host-visible memory, mapped */
struct img {
   VkImage image;
   VkDeviceMemory mem;
   VkImageView view;
   uint8_t *map;
   VkSubresourceLayout layout;
   VkDeviceSize size;
};

static int
make_image(struct img *im, VkFormat format, unsigned w, unsigned h, VkImageUsageFlags usage,
           VkImageLayout initial, const VkComponentMapping *swizzle)
{
   const VkImageCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                                   .format = format, .extent = { w, h, 1 }, .mipLevels = 1, .arrayLayers = 1,
                                   .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_LINEAR,
                                   .usage = usage, .initialLayout = initial };
   VkMemoryRequirements req;

   CHECK(vkCreateImage(dev, &ici, NULL, &im->image));
   vkGetImageMemoryRequirements(dev, im->image, &req);
   if (host_memory(&req, &im->mem))
      return 1;
   CHECK(vkBindImageMemory(dev, im->image, im->mem, 0));
   vkGetImageSubresourceLayout(dev, im->image, &(VkImageSubresource){ VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 },
                               &im->layout);
   im->size = req.size;
   CHECK(vkMapMemory(dev, im->mem, 0, VK_WHOLE_SIZE, 0, (void **)&im->map));
   const VkImageViewCreateInfo vci = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = im->image,
                                       .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format,
                                       .components = swizzle ? *swizzle : (VkComponentMapping){ 0 },
                                       .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
   CHECK(vkCreateImageView(dev, &vci, NULL, &im->view));
   return 0;
}

static void
free_image(struct img *im)
{
   vkDestroyImageView(dev, im->view, NULL);
   vkUnmapMemory(dev, im->mem);
   vkDestroyImage(dev, im->image, NULL);
   vkFreeMemory(dev, im->mem, NULL);
}

static int
flush(VkDeviceMemory mem, bool invalidate)
{
   const VkMappedMemoryRange range = { .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, .memory = mem,
                                       .size = VK_WHOLE_SIZE };

   if (invalidate)
      CHECK(vkInvalidateMappedMemoryRanges(dev, 1, &range));
   else
      CHECK(vkFlushMappedMemoryRanges(dev, 1, &range));
   return 0;
}

static int
run(const struct rung *r)
{
   const bool textured = r->tex != TEX_NONE;
   struct img target, tex = { 0 };
   VkSampler sampler = VK_NULL_HANDLE;
   VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
   VkDescriptorPool dpool = VK_NULL_HANDLE;
   VkDescriptorSet set = VK_NULL_HANDLE;

   cur = r->name;
   if (make_image(&target, VK_FORMAT_B8G8R8A8_UNORM, W, H, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                  VK_IMAGE_LAYOUT_UNDEFINED, NULL))
      return 1;
   memset(target.map, 0x5a, target.size);
   if (flush(target.mem, false))
      return 1;

   if (textured) {
      const VkComponentMapping bgra = { VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_R,
                                        VK_COMPONENT_SWIZZLE_A };

      if (make_image(&tex, VK_FORMAT_R8G8B8A8_UNORM, r->tw, r->th, VK_IMAGE_USAGE_SAMPLED_BIT,
                     VK_IMAGE_LAYOUT_PREINITIALIZED, r->swizzle ? &bgra : NULL))
         return 1;
      for (unsigned j = 0; j < r->th; j++) {
         for (unsigned i = 0; i < r->tw; i++)
            texel(r, i, j, tex.map + tex.layout.offset + j * tex.layout.rowPitch + 4 * i);
      }
      if (flush(tex.mem, false))
         return 1;
      const VkSamplerCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                        .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
                                        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                        .maxLod = 0.25f };
      CHECK(vkCreateSampler(dev, &sci, NULL, &sampler));
      const VkDescriptorSetLayoutBinding b = { .binding = 0,
                                               .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                               .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
      const VkDescriptorSetLayoutCreateInfo dlci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                                     .bindingCount = 1, .pBindings = &b };
      CHECK(vkCreateDescriptorSetLayout(dev, &dlci, NULL, &dsl));
      const VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 };
      const VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                                .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps };
      CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dpool));
      const VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                .descriptorPool = dpool, .descriptorSetCount = 1,
                                                .pSetLayouts = &dsl };
      CHECK(vkAllocateDescriptorSets(dev, &dai, &set));
      const VkDescriptorImageInfo dii = { sampler, tex.view, VK_IMAGE_LAYOUT_GENERAL };
      const VkWriteDescriptorSet wds = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
                                         .descriptorCount = 1,
                                         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                         .pImageInfo = &dii };
      vkUpdateDescriptorSets(dev, 1, &wds, 0, NULL);
   }

   /* render pass: clear, store; the layout ends GENERAL for the host read */
   const VkAttachmentDescription att = { .format = VK_FORMAT_B8G8R8A8_UNORM, .samples = VK_SAMPLE_COUNT_1_BIT,
                                         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                                         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                         .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                         .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                         .finalLayout = VK_IMAGE_LAYOUT_GENERAL };
   const VkAttachmentReference ref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   const VkSubpassDescription sp = { .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     .colorAttachmentCount = 1, .pColorAttachments = &ref };
   const VkRenderPassCreateInfo rci = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, .attachmentCount = 1,
                                        .pAttachments = &att, .subpassCount = 1, .pSubpasses = &sp };
   VkRenderPass rp;
   CHECK(vkCreateRenderPass(dev, &rci, NULL, &rp));
   const VkFramebufferCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = rp,
                                         .attachmentCount = 1, .pAttachments = &target.view, .width = W,
                                         .height = H, .layers = 1 };
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(dev, &fci, NULL, &fb));

   /* vertices: textured, the whole target (position, uv); solid, its left half (vec4 position) */
   const float tex_quad[16] = { -1, -1, 0, 0, 1, -1, 1, 0, -1, 1, 0, 1, 1, 1, 1, 1 };
   const float half_quad[16] = { -1, -1, 0, 1, 0, -1, 0, 1, -1, 1, 0, 1, 0, 1, 0, 1 };
   const VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = sizeof(tex_quad),
                                    .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT };
   VkBuffer vbuf;
   VkDeviceMemory vmem;
   VkMemoryRequirements vreq;
   void *vmap;
   CHECK(vkCreateBuffer(dev, &bci, NULL, &vbuf));
   vkGetBufferMemoryRequirements(dev, vbuf, &vreq);
   if (host_memory(&vreq, &vmem))
      return 1;
   CHECK(vkBindBufferMemory(dev, vbuf, vmem, 0));
   CHECK(vkMapMemory(dev, vmem, 0, VK_WHOLE_SIZE, 0, &vmap));
   memcpy(vmap, textured ? tex_quad : half_quad, sizeof(tex_quad));
   if (flush(vmem, false))
      return 1;

   /* pipeline */
   VkShaderModuleCreateInfo smi = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
   VkShaderModule vsm, fsm;
   smi.codeSize = textured ? sizeof(spv_uv_vert) : sizeof(spv_pass_vert);
   smi.pCode = textured ? spv_uv_vert : spv_pass_vert;
   CHECK(vkCreateShaderModule(dev, &smi, NULL, &vsm));
   smi.codeSize = textured ? sizeof(spv_tex_frag) : sizeof(spv_solid_frag);
   smi.pCode = textured ? spv_tex_frag : spv_solid_frag;
   CHECK(vkCreateShaderModule(dev, &smi, NULL, &fsm));
   const VkPipelineLayoutCreateInfo lci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                            .setLayoutCount = textured ? 1 : 0, .pSetLayouts = &dsl };
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(dev, &lci, NULL, &layout));
   const VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vsm, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = fsm, .pName = "main" },
   };
   const VkVertexInputBindingDescription vib = { 0, 16, VK_VERTEX_INPUT_RATE_VERTEX };
   const VkVertexInputAttributeDescription tex_attr[2] = { { 0, 0, VK_FORMAT_R32G32_SFLOAT, 0 },
                                                           { 1, 0, VK_FORMAT_R32G32_SFLOAT, 8 } };
   const VkVertexInputAttributeDescription pos_attr = { 0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0 };
   const VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &vib, .vertexAttributeDescriptionCount = textured ? 2 : 1,
      .pVertexAttributeDescriptions = textured ? tex_attr : &pos_attr };
   const VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP };
   const VkPipelineViewportStateCreateInfo vps = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                                   .viewportCount = 1, .scissorCount = 1 };
   const VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE, .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f };
   const VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
   const VkPipelineColorBlendAttachmentState cba = {
      .blendEnable = r->blend, .srcColorBlendFactor = r->src_rgb, .dstColorBlendFactor = r->dst_rgb,
      .colorBlendOp = r->op_rgb, .srcAlphaBlendFactor = r->src_a, .dstAlphaBlendFactor = r->dst_a,
      .alphaBlendOp = r->op_a, .colorWriteMask = 0xf };
   const VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1,
      .pAttachments = &cba };
   const VkDynamicState dyns[3] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                    VK_DYNAMIC_STATE_BLEND_CONSTANTS };
   const VkPipelineDynamicStateCreateInfo ds = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                                 .dynamicStateCount = r->constants ? 3 : 2,
                                                 .pDynamicStates = dyns };
   const VkGraphicsPipelineCreateInfo gci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = 2, .pStages = stages,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vps,
      .pRasterizationState = &rs, .pMultisampleState = &ms, .pColorBlendState = &cb, .pDynamicState = &ds,
      .layout = layout, .renderPass = rp };
   VkPipeline pipe;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gci, NULL, &pipe));

   /* record and submit */
   const VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                         .queueFamilyIndex = queue_family };
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(dev, &pci, NULL, &pool));
   const VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                             .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                             .commandBufferCount = 1 };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
   const VkCommandBufferBeginInfo cbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
   CHECK(vkBeginCommandBuffer(cmd, &cbi));
   if (textured) {
      const VkImageMemoryBarrier imb = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                         .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
                                         .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                                         .oldLayout = VK_IMAGE_LAYOUT_PREINITIALIZED,
                                         .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = tex.image,
                                         .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0,
                           NULL, 1, &imb);
   }
   const VkClearValue cv = { .color.float32 = { CLEAR[0], CLEAR[1], CLEAR[2], CLEAR[3] } };
   const VkRenderPassBeginInfo rbi = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = rp,
                                       .framebuffer = fb, .renderArea = { { 0, 0 }, { W, H } },
                                       .clearValueCount = 1, .pClearValues = &cv };
   vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   if (textured)
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, NULL);
   const VkDeviceSize zero = 0;
   vkCmdBindVertexBuffers(cmd, 0, 1, &vbuf, &zero);
   const VkViewport vpt = { 0, 0, W, H, 0, 1 };
   const VkRect2D sc = { { 0, 0 }, { W, H } };
   vkCmdSetViewport(cmd, 0, 1, &vpt);
   vkCmdSetScissor(cmd, 0, 1, &sc);
   if (r->constants)
      vkCmdSetBlendConstants(cmd, CONSTANTS);
   vkCmdDraw(cmd, 4, 1, 0, 0);
   vkCmdEndRenderPass(cmd);
   CHECK(vkEndCommandBuffer(cmd));
   const VkFenceCreateInfo fnci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
   VkFence fence;
   CHECK(vkCreateFence(dev, &fnci, NULL, &fence));
   const VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                             .pCommandBuffers = &cmd };
   CHECK(vkQueueSubmit(queue, 1, &si, fence));
   CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 5000000000ull));
   if (flush(target.mem, true))
      return 1;

   /* compare: expected RGBA per pixel, the target stores B, G, R, A */
   uint8_t clear8[4];
   double src[4];
   unsigned bad = 0, fx = 0, fy = 0, tol = r->blend ? 1 : 0, maxdiff = 0;
   uint8_t got_f[4] = { 0 }, want_f[4] = { 0 };

   for (unsigned c = 0; c < 4; c++) {
      clear8[c] = unorm8(CLEAR[c]);
      src[c] = SRC[c];
   }
   for (unsigned y = 0; verify && y < H; y++) {
      for (unsigned x = 0; x < W; x++) {
         const uint8_t *p = target.map + target.layout.offset + y * target.layout.rowPitch + 4 * x;
         const uint8_t got[4] = { p[2], p[1], p[0], p[3] };
         uint8_t want[4];

         if (textured) {
            uint8_t t[4];
            const unsigned i = (unsigned)(((x + 0.5) / W) * r->tw), j = (unsigned)(((y + 0.5) / H) * r->th);

            texel(r, i, j, t);
            if (r->swizzle) {
               const uint8_t s0 = t[0];
               t[0] = t[2];
               t[2] = s0;
            }
            if (r->blend) {
               double s[4];
               for (unsigned c = 0; c < 4; c++)
                  s[c] = t[c] / 255.0;
               blend_ref(r, s, clear8, want);
            } else {
               memcpy(want, t, 4);
            }
         } else if (x < W / 2) {
            blend_ref(r, src, clear8, want);
         } else {
            memcpy(want, clear8, 4);
         }
         unsigned d = 0;
         for (unsigned c = 0; c < 4; c++) {
            const unsigned e = got[c] > want[c] ? got[c] - want[c] : want[c] - got[c];
            d = e > d ? e : d;
         }
         maxdiff = d > maxdiff ? d : maxdiff;
         if (d > tol) {
            if (!bad++) {
               fx = x, fy = y;
               memcpy(got_f, got, 4);
               memcpy(want_f, want, 4);
            }
         }
      }
   }
   if (!verify)
      printf("TEX-BLEND %s PASS: submitted and retired (not compared)\n", r->name);
   else if (bad)
      printf("TEX-BLEND %s FAIL: %u of %u pixels differ (tolerance %u), first at %u,%u: RGBA %u %u %u %u, "
             "expected %u %u %u %u\n",
             r->name, bad, W * H, tol, fx, fy, got_f[0], got_f[1], got_f[2], got_f[3], want_f[0], want_f[1],
             want_f[2], want_f[3]);
   else
      printf("TEX-BLEND %s PASS: %u pixels, largest difference %u (tolerance %u)\n", r->name, W * H, maxdiff,
             tol);

   vkDestroyFence(dev, fence, NULL);
   vkFreeCommandBuffers(dev, pool, 1, &cmd);
   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   vkDestroyShaderModule(dev, vsm, NULL);
   vkDestroyShaderModule(dev, fsm, NULL);
   vkUnmapMemory(dev, vmem);
   vkDestroyBuffer(dev, vbuf, NULL);
   vkFreeMemory(dev, vmem, NULL);
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyRenderPass(dev, rp, NULL);
   if (textured) {
      vkDestroyDescriptorPool(dev, dpool, NULL);
      vkDestroyDescriptorSetLayout(dev, dsl, NULL);
      vkDestroySampler(dev, sampler, NULL);
      free_image(&tex);
   }
   free_image(&target);
   return verify && bad ? 1 : 0;
}

int
main(int argc, char **argv)
{
   unsigned device_index = 0, nsel = 0, pass = 0, ran = 0;
   const char *sel[ARRAY_LEN(rungs)];

   setvbuf(stdout, NULL, _IOLBF, 0);
   for (int a = 1; a < argc; a++) {
      if (!strcmp(argv[a], "--list")) {
         for (unsigned i = 0; i < ARRAY_LEN(rungs); i++)
            printf("%s\n", rungs[i].name);
         return 0;
      } else if (!strcmp(argv[a], "--device") && a + 1 < argc) {
         device_index = (unsigned)atoi(argv[++a]);
      } else if (!strcmp(argv[a], "--no-verify")) {
         verify = false;
      } else if (argv[a][0] != '-' && nsel < ARRAY_LEN(sel)) {
         sel[nsel++] = argv[a];
      } else {
         fprintf(stderr, "usage: %s [--list] [--device N] [--no-verify] [RUNG...]\n", argv[0]);
         return 2;
      }
   }

   const VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "tex-blend",
                                   .apiVersion = VK_API_VERSION_1_0 };
   const VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                      .pApplicationInfo = &app };
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 0;
   CHECK(vkEnumeratePhysicalDevices(inst, &n, NULL));
   VkPhysicalDevice pds[16];
   n = n > 16 ? 16 : n;
   CHECK(vkEnumeratePhysicalDevices(inst, &n, pds));
   if (device_index >= n) {
      printf("TEX-BLEND RESULT FAIL: no Vulkan device %u (%u found)\n", device_index, n);
      return 1;
   }
   pd = pds[device_index];
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("TEX-BLEND device %u: %s, Vulkan %u.%u, driver %#x\n", device_index, props.deviceName,
          VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion), props.driverVersion);

   VkQueueFamilyProperties qf[8];
   uint32_t nq = 8;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf);
   queue_family = UINT32_MAX;
   for (uint32_t i = 0; i < nq && queue_family == UINT32_MAX; i++) {
      if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
         queue_family = i;
   }
   if (queue_family == UINT32_MAX) {
      printf("TEX-BLEND RESULT FAIL: no graphics queue\n");
      return 1;
   }
   const float prio = 1.0f;
   const VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                         .queueFamilyIndex = queue_family, .queueCount = 1,
                                         .pQueuePriorities = &prio };
   const VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
                                    .pQueueCreateInfos = &qci };
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   vkGetDeviceQueue(dev, queue_family, 0, &queue);

   for (unsigned i = 0; i < ARRAY_LEN(rungs); i++) {
      bool want = !nsel;
      for (unsigned k = 0; k < nsel; k++)
         want |= !strcmp(sel[k], rungs[i].name);
      if (!want)
         continue;
      ran++;
      if (!run(&rungs[i]))
         pass++;
   }
   for (unsigned k = 0; k < nsel; k++) {
      bool known = false;
      for (unsigned i = 0; i < ARRAY_LEN(rungs); i++)
         known |= !strcmp(sel[k], rungs[i].name);
      if (!known) {
         printf("TEX-BLEND %s FAIL: no such rung (--list)\n", sel[k]);
         ran++;
      }
   }
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   printf("TEX-BLEND RESULT %s %u of %u\n", ran && pass == ran ? "PASS" : "FAIL", pass, ran);
   return ran && pass == ran ? 0 : 1;
}
