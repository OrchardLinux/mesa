/* SPDX-License-Identifier: MIT */
/* ds-sample-vk: sampled depth and stencil (ES 3.0's depth textures, GL_DEPTH_COMPONENT32F, shadow samplers),
 * plain Vulkan 1.0, checked against a CPU model. Two render jobs per case on a 64x64 linear B8G8R8A8 target:
 *   job 0  renders the depth/stencil image: a full quad at z=.601 (stencil 3, Replace), then a pixel-aligned rect
 *          16,24-48,56 at z=.201 (stencil 9), Less + write; depth/stencil stored, final layout DEPTH_STENCIL_READ_ONLY
 *   job 1  a full quad samples it with a nearest sampler and writes R = the result (G=B=0, A=1)
 * Cases (D32F, or the depth/stencil aspects of D32F_S8):
 *   raw, raw8         sampler2D, the depth as is: R = round(255 * d): 51 inside, 153 outside
 *   cmp, cmp8         sampler2DShadow, compareOp LESS against reference .4 (Vulkan: ref < d): 0 inside, 255 outside
 *   stencil8          usampler2D on the stencil aspect: R = s: 9 inside, 3 outside
 * The depths and the stencil output sit a quarter step above the integers (51.255, 153.255, 9.25), so the
 * colour conversion's rounding or truncation gives the same bytes.
 *
 *   ds-sample-vk [--offline] [--case NAME[,NAME...]]
 *
 * Prints "device=... sampled D32F=x D32F_S8=y filter_linear=a/b", one line per job, "case=... result=EXACT bad=0" or
 * MISMATCH, then result=PASS. Exit 0 all exact, 1 a pixel mismatch, 3 setup error or a format not sampled
 * (result=NOT-OFFERED), 4 a job refused or failed (-8 refused, -4 device lost, 2 fence timeout). --offline (a
 * driver that executes nothing, no pixels): every job must be accepted (result=ACCEPTED).
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "color.frag.h"
#include "color.vert.h"
#include "ds-cmp.frag.h"
#include "ds-raw.frag.h"
#include "ds-stencil.frag.h"
#include "uv.vert.h"

#define CHECK(x)                                                                                   \
	do {                                                                                       \
		VkResult r_ = (x);                                                                 \
		if (r_ != VK_SUCCESS) {                                                            \
			fprintf(stderr, "ds-sample-vk: %s = %d (line %d)\n", #x, r_, __LINE__);    \
			exit(3);                                                                   \
		}                                                                                  \
	} while (0)

#define N 64u
#define RX0 16
#define RY0 24
#define RX1 48
#define RY1 56
#define Z_IN .201f
#define Z_OUT .601f
#define S_IN 9u
#define S_OUT 3u

enum kind { RAW, CMP, STENCIL };
struct tcase {
	const char *name;
	VkFormat format;
	enum kind kind;
	uint8_t want_in, want_out;
};
static const struct tcase cases[] = {
	{"raw", VK_FORMAT_D32_SFLOAT, RAW, 51, 153},
	{"cmp", VK_FORMAT_D32_SFLOAT, CMP, 0, 255},
	{"raw8", VK_FORMAT_D32_SFLOAT_S8_UINT, RAW, 51, 153},
	{"cmp8", VK_FORMAT_D32_SFLOAT_S8_UINT, CMP, 0, 255},
	{"stencil8", VK_FORMAT_D32_SFLOAT_S8_UINT, STENCIL, S_IN, S_OUT},
};

static VkPhysicalDevice phys;
static VkDevice dev;
static VkQueue queue;
static VkCommandPool pool;
static VkPhysicalDeviceMemoryProperties memprops;
static bool offline;

static uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want)
{
	for (uint32_t i = 0; i < memprops.memoryTypeCount; i++)
		if ((bits & (1u << i)) && (memprops.memoryTypes[i].propertyFlags & want) == want)
			return i;
	for (uint32_t i = 0; i < memprops.memoryTypeCount; i++)
		if (bits & (1u << i))
			return i;
	return 0;
}

static VkDeviceMemory alloc_bind_image(VkImage img, VkMemoryPropertyFlags want)
{
	VkMemoryRequirements r;
	VkDeviceMemory m;
	vkGetImageMemoryRequirements(dev, img, &r);
	VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = r.size,
				   .memoryTypeIndex = mem_type(r.memoryTypeBits, want)};
	CHECK(vkAllocateMemory(dev, &ai, NULL, &m));
	CHECK(vkBindImageMemory(dev, img, m, 0));
	return m;
}

static VkImageView make_view(VkImage img, VkFormat f, VkImageAspectFlags aspects)
{
	VkImageView v;
	VkImageViewCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img,
				    .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = f,
				    .subresourceRange = {aspects, 0, 1, 0, 1}};
	CHECK(vkCreateImageView(dev, &ci, NULL, &v));
	return v;
}

static VkBuffer make_vbuf(const float *data, size_t bytes, VkDeviceMemory *mem)
{
	VkBuffer b;
	VkMemoryRequirements r;
	void *p;
	VkBufferCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes,
				 .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT};
	CHECK(vkCreateBuffer(dev, &ci, NULL, &b));
	vkGetBufferMemoryRequirements(dev, b, &r);
	VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = r.size,
				   .memoryTypeIndex = mem_type(r.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
								     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
	CHECK(vkAllocateMemory(dev, &ai, NULL, mem));
	CHECK(vkBindBufferMemory(dev, b, *mem, 0));
	CHECK(vkMapMemory(dev, *mem, 0, bytes, 0, &p));
	memcpy(p, data, bytes);
	vkUnmapMemory(dev, *mem);
	return b;
}

static VkShaderModule shader(const uint32_t *code, size_t bytes)
{
	VkShaderModule m;
	VkShaderModuleCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = bytes,
				       .pCode = code};
	CHECK(vkCreateShaderModule(dev, &ci, NULL, &m));
	return m;
}

/* depth: job 0's render pass (colour + depth/stencil), else job 1's (colour only) */
static VkRenderPass make_pass(VkFormat ds)
{
	VkRenderPass rp;
	VkAttachmentDescription a[2] = {
		{.format = VK_FORMAT_B8G8R8A8_UNORM, .samples = VK_SAMPLE_COUNT_1_BIT, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		 .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
		 .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		 .finalLayout = VK_IMAGE_LAYOUT_GENERAL},
		{.format = ds, .samples = VK_SAMPLE_COUNT_1_BIT, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		 .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		 .stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		 .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL}};
	VkAttachmentReference cref = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
	VkAttachmentReference dref = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
	VkSubpassDescription sp = {.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS, .colorAttachmentCount = 1,
				   .pColorAttachments = &cref, .pDepthStencilAttachment = ds ? &dref : NULL};
	VkSubpassDependency dep = {.srcSubpass = 0, .dstSubpass = VK_SUBPASS_EXTERNAL,
				   .srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
						   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				   .dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
				   .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
						    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				   .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_HOST_READ_BIT};
	VkRenderPassCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, .attachmentCount = ds ? 2 : 1,
				     .pAttachments = a, .subpassCount = 1, .pSubpasses = &sp, .dependencyCount = 1,
				     .pDependencies = &dep};
	CHECK(vkCreateRenderPass(dev, &ci, NULL, &rp));
	return rp;
}

/* job 0: stencil reference `ref`, Less + write; job 1 (ref < 0): the sampling program, no depth/stencil */
static VkPipeline make_pipeline(VkRenderPass rp, VkPipelineLayout layout, VkShaderModule vs, VkShaderModule fs,
				bool uv, bool stencil, int ref)
{
	VkPipeline p;
	VkPipelineShaderStageCreateInfo st[2] = {
		{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
		 .module = vs, .pName = "main"},
		{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
		 .module = fs, .pName = "main"}};
	VkVertexInputBindingDescription b = {0, uv ? 16 : 32, VK_VERTEX_INPUT_RATE_VERTEX};
	VkVertexInputAttributeDescription a[2] = {
		{.location = 0, .format = uv ? VK_FORMAT_R32G32_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT},
		{.location = 1, .format = uv ? VK_FORMAT_R32G32_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT,
		 .offset = uv ? 8 : 16}};
	VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
						   .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &b,
						   .vertexAttributeDescriptionCount = 2, .pVertexAttributeDescriptions = a};
	VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
						     .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
	VkViewport vp = {0, 0, N, N, 0, 1};
	VkRect2D sc = {{0, 0}, {N, N}};
	VkPipelineViewportStateCreateInfo vps = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
						 .viewportCount = 1, .pViewports = &vp, .scissorCount = 1, .pScissors = &sc};
	VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
						     .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
						     .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1};
	VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
						   .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
	VkStencilOpState so = {.failOp = VK_STENCIL_OP_KEEP, .passOp = VK_STENCIL_OP_REPLACE,
			       .depthFailOp = VK_STENCIL_OP_KEEP, .compareOp = VK_COMPARE_OP_ALWAYS, .compareMask = 0xff,
			       .writeMask = 0xff, .reference = ref < 0 ? 0 : (uint32_t)ref};
	VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
						    .depthTestEnable = ref >= 0, .depthWriteEnable = ref >= 0,
						    .depthCompareOp = VK_COMPARE_OP_LESS,
						    .stencilTestEnable = ref >= 0 && stencil, .front = so, .back = so};
	VkPipelineColorBlendAttachmentState cba = {.colorWriteMask = 0xf};
	VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
						  .attachmentCount = 1, .pAttachments = &cba};
	VkGraphicsPipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = 2,
					   .pStages = st, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
					   .pViewportState = &vps, .pRasterizationState = &rs, .pMultisampleState = &ms,
					   .pDepthStencilState = ref >= 0 ? &ds : NULL, .pColorBlendState = &cb,
					   .layout = layout, .renderPass = rp};
	CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &ci, NULL, &p));
	return p;
}

/* one job: record, submit, wait. Returns the VkResult of the submit or the wait. */
static VkResult run_job(VkCommandBuffer cmd)
{
	VkFence f;
	VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	CHECK(vkCreateFence(dev, &fci, NULL, &f));
	VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd};
	VkResult r = vkQueueSubmit(queue, 1, &si, f);
	if (r == VK_SUCCESS)
		r = vkWaitForFences(dev, 1, &f, VK_TRUE, 5000000000ull);
	vkDestroyFence(dev, f, NULL);
	return r;
}

/* x0..y1 in pixels to two triangles of vec4 position + vec4 colour at depth z */
static void quad_color(float *v, int x0, int y0, int x1, int y1, float z)
{
	const float X0 = x0 * 2.f / N - 1, X1 = x1 * 2.f / N - 1, Y0 = y0 * 2.f / N - 1, Y1 = y1 * 2.f / N - 1;
	const float c[2][2] = {{X0, Y0}, {X1, Y1}};
	const int idx[6][2] = {{0, 0}, {1, 0}, {0, 1}, {1, 0}, {1, 1}, {0, 1}};
	for (int i = 0; i < 6; i++) {
		float *o = v + 8 * i;
		o[0] = c[idx[i][0]][0], o[1] = c[idx[i][1]][1], o[2] = z, o[3] = 1;
		o[4] = 1, o[5] = 0, o[6] = 0, o[7] = 1;
	}
}

static int run_case(const struct tcase *k)
{
	const bool has_stencil = k->format == VK_FORMAT_D32_SFLOAT_S8_UINT;
	VkImage cimg, dimg;
	VkDeviceMemory cmem, dmem, vmem0, vmem1;
	void *cmap = NULL;

	/* the colour target: linear, host-visible, read back */
	VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
				.format = VK_FORMAT_B8G8R8A8_UNORM, .extent = {N, N, 1}, .mipLevels = 1, .arrayLayers = 1,
				.samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_LINEAR,
				.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
				.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
	CHECK(vkCreateImage(dev, &ci, NULL, &cimg));
	cmem = alloc_bind_image(cimg, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	VkImageSubresource sub = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
	VkSubresourceLayout lay;
	vkGetImageSubresourceLayout(dev, cimg, &sub, &lay);
	CHECK(vkMapMemory(dev, cmem, 0, VK_WHOLE_SIZE, 0, &cmap));
	VkImageView cview = make_view(cimg, ci.format, VK_IMAGE_ASPECT_COLOR_BIT);

	/* the depth/stencil image: an attachment of job 0, sampled by job 1 */
	ci.format = k->format;
	ci.tiling = VK_IMAGE_TILING_OPTIMAL;
	ci.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	CHECK(vkCreateImage(dev, &ci, NULL, &dimg));
	dmem = alloc_bind_image(dimg, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	(void)dmem; /* freed with the process */
	const VkImageAspectFlags all = VK_IMAGE_ASPECT_DEPTH_BIT | (has_stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
	VkImageView dview = make_view(dimg, k->format, all);
	VkImageView sview = make_view(dimg, k->format,
				      k->kind == STENCIL ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT);

	VkRenderPass rp0 = make_pass(k->format), rp1 = make_pass(VK_FORMAT_UNDEFINED);
	VkFramebuffer fb0, fb1;
	VkImageView att0[2] = {cview, dview};
	VkFramebufferCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = rp0,
				       .attachmentCount = 2, .pAttachments = att0, .width = N, .height = N, .layers = 1};
	CHECK(vkCreateFramebuffer(dev, &fci, NULL, &fb0));
	fci.renderPass = rp1, fci.attachmentCount = 1;
	CHECK(vkCreateFramebuffer(dev, &fci, NULL, &fb1));

	/* the sampler and its descriptor */
	VkSampler smp;
	VkSamplerCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST,
				   .minFilter = VK_FILTER_NEAREST, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
				   .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
				   .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
				   .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
				   .compareEnable = k->kind == CMP, .compareOp = VK_COMPARE_OP_LESS,
				   .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK};
	CHECK(vkCreateSampler(dev, &sci, NULL, &smp));
	VkDescriptorSetLayoutBinding bind = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
					     VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
	VkDescriptorSetLayout dsl;
	VkDescriptorSetLayoutCreateInfo dslci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
						 .bindingCount = 1, .pBindings = &bind};
	CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
	VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
	VkDescriptorPool dp;
	VkDescriptorPoolCreateInfo dpci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1,
					   .poolSizeCount = 1, .pPoolSizes = &ps};
	CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
	VkDescriptorSet set;
	VkDescriptorSetAllocateInfo dsai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dp,
					    .descriptorSetCount = 1, .pSetLayouts = &dsl};
	CHECK(vkAllocateDescriptorSets(dev, &dsai, &set));
	VkDescriptorImageInfo dii = {smp, sview, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
	VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .descriptorCount = 1,
				  .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &dii};
	vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);

	VkPipelineLayout pl0, pl1;
	VkPipelineLayoutCreateInfo plci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl0));
	plci.setLayoutCount = 1, plci.pSetLayouts = &dsl;
	CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl1));
	VkShaderModule cvs = shader(spv_color_vert, sizeof(spv_color_vert)),
		       cfs = shader(spv_color_frag, sizeof(spv_color_frag)),
		       uvs = shader(spv_uv_vert, sizeof(spv_uv_vert));
	VkShaderModule sfs = k->kind == RAW   ? shader(spv_ds_raw_frag, sizeof(spv_ds_raw_frag))
			     : k->kind == CMP ? shader(spv_ds_cmp_frag, sizeof(spv_ds_cmp_frag))
					      : shader(spv_ds_stencil_frag, sizeof(spv_ds_stencil_frag));
	VkPipeline p_out = make_pipeline(rp0, pl0, cvs, cfs, false, has_stencil, S_OUT);
	VkPipeline p_in = make_pipeline(rp0, pl0, cvs, cfs, false, has_stencil, S_IN);
	VkPipeline p_smp = make_pipeline(rp1, pl1, uvs, sfs, true, false, -1);

	float v0[2 * 6 * 8];
	quad_color(v0, 0, 0, N, N, Z_OUT);
	quad_color(v0 + 48, RX0, RY0, RX1, RY1, Z_IN);
	const float v1[6 * 4] = {-1, -1, 0, 0, 1, -1, 1, 0, -1, 1, 0, 1, 1, -1, 1, 0, 1, 1, 1, 1, -1, 1, 0, 1};
	VkBuffer vb0 = make_vbuf(v0, sizeof(v0), &vmem0), vb1 = make_vbuf(v1, sizeof(v1), &vmem1);
	VkDeviceSize zero = 0;

	VkCommandBuffer cmd[2];
	VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
					   .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 2};
	CHECK(vkAllocateCommandBuffers(dev, &cai, cmd));
	VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	VkClearValue clr[2] = {{.color = {{0, 0, 0, 1}}}, {.depthStencil = {1.0f, 0}}};

	CHECK(vkBeginCommandBuffer(cmd[0], &bi));
	VkRenderPassBeginInfo rbi = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = rp0,
				     .framebuffer = fb0, .renderArea = {{0, 0}, {N, N}}, .clearValueCount = 2,
				     .pClearValues = clr};
	vkCmdBeginRenderPass(cmd[0], &rbi, VK_SUBPASS_CONTENTS_INLINE);
	vkCmdBindVertexBuffers(cmd[0], 0, 1, &vb0, &zero);
	vkCmdBindPipeline(cmd[0], VK_PIPELINE_BIND_POINT_GRAPHICS, p_out);
	vkCmdDraw(cmd[0], 6, 1, 0, 0);
	vkCmdBindPipeline(cmd[0], VK_PIPELINE_BIND_POINT_GRAPHICS, p_in);
	vkCmdDraw(cmd[0], 6, 1, 6, 0);
	vkCmdEndRenderPass(cmd[0]);
	CHECK(vkEndCommandBuffer(cmd[0]));

	CHECK(vkBeginCommandBuffer(cmd[1], &bi));
	rbi.renderPass = rp1, rbi.framebuffer = fb1, rbi.clearValueCount = 1;
	vkCmdBeginRenderPass(cmd[1], &rbi, VK_SUBPASS_CONTENTS_INLINE);
	vkCmdBindVertexBuffers(cmd[1], 0, 1, &vb1, &zero);
	vkCmdBindPipeline(cmd[1], VK_PIPELINE_BIND_POINT_GRAPHICS, p_smp);
	vkCmdBindDescriptorSets(cmd[1], VK_PIPELINE_BIND_POINT_GRAPHICS, pl1, 0, 1, &set, 0, NULL);
	vkCmdDraw(cmd[1], 6, 1, 0, 0);
	vkCmdEndRenderPass(cmd[1]);
	CHECK(vkEndCommandBuffer(cmd[1]));

	for (int j = 0; j < 2; j++) {
		VkResult r = run_job(cmd[j]);
		if (r != VK_SUCCESS) {
			printf("case=%s job=%d result=FAILED vk=%d\n", k->name, j, r);
			return 4;
		}
		printf("case=%s job=%d result=%s\n", k->name, j, offline ? "ACCEPTED" : "DONE");
	}
	if (offline) {
		printf("case=%s result=ACCEPTED\n", k->name);
		return 0;
	}
	unsigned bad = 0, first = ~0u;
	for (uint32_t y = 0; y < N; y++)
		for (uint32_t x = 0; x < N; x++) {
			const uint8_t *p = (const uint8_t *)cmap + y * lay.rowPitch + 4 * x;
			const bool in = x >= RX0 && x < RX1 && y >= RY0 && y < RY1;
			const uint8_t want_r = in ? k->want_in : k->want_out;
			/* B8G8R8A8: B, G, R, A */
			if (p[0] != 0 || p[1] != 0 || p[2] != want_r || p[3] != 255) {
				if (!bad++)
					first = y * N + x;
			}
		}
	const uint8_t *fp = bad ? (const uint8_t *)cmap + (first / N) * lay.rowPitch + 4 * (first % N) : NULL;
	printf("case=%s result=%s bad=%u", k->name, bad ? "MISMATCH" : "EXACT", bad);
	if (bad)
		printf(" first=%u,%u bgra=%u,%u,%u,%u", first % N, first / N, fp[0], fp[1], fp[2], fp[3]);
	printf(" want_in=%u want_out=%u\n", k->want_in, k->want_out);
	return bad ? 1 : 0;
}

int main(int argc, char **argv)
{
	const char *only = NULL;
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--offline"))
			offline = true;
		else if (!strcmp(argv[i], "--case") && i + 1 < argc)
			only = argv[++i];
	}
	VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_0};
	VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
	VkInstance inst;
	CHECK(vkCreateInstance(&ici, NULL, &inst));
	uint32_t n = 1;
	VkResult er = vkEnumeratePhysicalDevices(inst, &n, &phys);
	if ((er != VK_SUCCESS && er != VK_INCOMPLETE) || !n) {
		fprintf(stderr, "ds-sample-vk: no device\n");
		return 3;
	}
	vkGetPhysicalDeviceMemoryProperties(phys, &memprops);
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(phys, &props);
	VkFormatProperties f32, f32s8;
	vkGetPhysicalDeviceFormatProperties(phys, VK_FORMAT_D32_SFLOAT, &f32);
	vkGetPhysicalDeviceFormatProperties(phys, VK_FORMAT_D32_SFLOAT_S8_UINT, &f32s8);
	const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
	const bool s32 = (f32.optimalTilingFeatures & need) == need, s32s8 = (f32s8.optimalTilingFeatures & need) == need;
	printf("device=%s sampled D32F=%d D32F_S8=%d filter_linear=%d/%d\n", props.deviceName, s32, s32s8,
	       !!(f32.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT),
	       !!(f32s8.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT));

	float prio = 1;
	VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1,
				       .pQueuePriorities = &prio};
	VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
				  .pQueueCreateInfos = &qci};
	CHECK(vkCreateDevice(phys, &dci, NULL, &dev));
	vkGetDeviceQueue(dev, 0, 0, &queue);
	VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
	CHECK(vkCreateCommandPool(dev, &pci, NULL, &pool));

	int rc = 0, ran = 0;
	for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
		const struct tcase *k = &cases[i];
		if (only) {
			const char *s = strstr(only, k->name);
			const size_t l = strlen(k->name);
			if (!s || (s != only && s[-1] != ',') || (s[l] && s[l] != ','))
				continue;
		}
		if (!(k->format == VK_FORMAT_D32_SFLOAT ? s32 : s32s8)) {
			printf("case=%s result=NOT-OFFERED (%s not a sampled depth attachment)\n", k->name,
			       k->format == VK_FORMAT_D32_SFLOAT ? "D32F" : "D32F_S8");
			if (!rc)
				rc = 3;
			continue;
		}
		ran++;
		const int r = run_case(k);
		if (r > rc)
			rc = r;
	}
	if (rc == 3)
		printf("result=NOT-OFFERED\n");
	else
		printf("result=%s\n", rc ? "FAIL" : offline ? "ACCEPTED" : "PASS");
	return ran ? rc : 3;
}
