/* SPDX-License-Identifier: MIT */
/* tt-vk: 2D array and cube textures (and NPOT mip chains), plain Vulkan 1.0, checked against a CPU model.
 * Ported from the Apple A9 (G5P) Linux port's texture-types fixture, cut to the cases the G9P ES 3.0 option
 * implements; every case below is exact on A9 hardware (cube, npot-lod and arr-mip after the fixes noted).
 *
 * One device; every case is one render job (one submit + fence wait) into a 64x64 BGRA8 target, read back and
 * compared per pixel with a host model. Textures are filled by vkCmdCopyBufferToImage into allocations padded
 * by 256 KiB + 3x the image, so a misread layer/level stride stays inside the texture's own allocation. Nearest
 * filtering and mip-nearest throughout, so every pixel has one exact texel:
 *   one          the one-target pass (red triangle), no texture: run it first
 *   arr          4x4 RGBA8 2-layer array, layer 1 (texel x*60 | y*60 << 8 | L*40 << 16)
 *   arr-layer    the array, layer = round(u * 2) clamped (left quarter layer 0, else layer 1)
 *   cube         4x4 cube, direction (u, -1, v): face 3 (-Y). On the A9 the first run read face 0 at row 0
 *                (blue 0): the direction went to the sampler as 2D coordinates without the cube op
 *   npot npot-lod  12x10 4-level 2D chain: level 0 magnified, textureLod 2 (3x2). On the A9 npot-lod first
 *                failed: each level is twiddled over its own extent padded to powers of two
 *   arr-mip      8x8 2-layer 4-level array, textureLod(layer 1, level 1): the mipmapped layer stride. On the
 *                A9 it first failed: the layer stride is the chain rounded to 64 bytes (16 KiB past 16 KiB)
 * Level L texel (x, y) of a chain is (L*40 + 20 + layer*100, x, y, 255). Prints one line per case
 * (`case=NAME result=EXACT|CLOSE|BAD bad=N maxdiff=D`) and result=. --out DIR keeps DIR/<case>.raw.
 * Exit 0 all good, 1 a pixel mismatch, 3 setup error, 4 a job was refused or failed (VkResult: -8
 * refused, -4 device lost, 2 fence timeout); a refused or failed case stops the run. --offline (a driver that
 * executes nothing, no pixels): only that every job is accepted.
 *
 *   tt-vk [--offline] [--out DIR] [--case NAME[,NAME...]]
 */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "pass_vert.h"
#include "solid_frag.h"
#include "tex_frag.h"
#include "tt_arr_frag.h"
#include "tt_arr_lod_frag.h"
#include "tt_arr_lod1_frag.h"
#include "tt_cube_frag.h"
#include "tt_lod_frag.h"
#include "uv_vert.h"

#define CHECK(x)                                                                         \
	do {                                                                             \
		VkResult r_ = (x);                                                       \
		if (r_ != VK_SUCCESS) {                                                  \
			fprintf(stderr, "tt-vk: %s = %d (line %d)\n", #x, r_, __LINE__); \
			exit(3);                                                         \
		}                                                                        \
	} while (0)
#define N 64
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#define SHADER(n) n, sizeof(n)

static VkDevice dev;
static VkQueue queue;
static VkCommandPool pool;
static VkPhysicalDeviceMemoryProperties memprops;
static bool offline;
static bool skip[N * N];
static const char *outdir;

static uint32_t mem_type(uint32_t bits)
{
	for (uint32_t i = 0; i < memprops.memoryTypeCount; i++)
		if ((bits & (1u << i)) && (memprops.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
			return i;
	fprintf(stderr, "tt-vk: no host-visible memory\n");
	exit(3);
}

static VkDeviceMemory alloc(VkMemoryRequirements req, VkDeviceSize pad)
{
	VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = req.size + pad, .memoryTypeIndex = mem_type(req.memoryTypeBits)};
	VkDeviceMemory m;
	CHECK(vkAllocateMemory(dev, &ai, NULL, &m));
	return m;
}

static VkBuffer make_buffer(const void *data, size_t size, VkBufferUsageFlags usage)
{
	VkBuffer b;
	void *p;
	VkBufferCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage};
	CHECK(vkCreateBuffer(dev, &ci, NULL, &b));
	VkMemoryRequirements req;
	vkGetBufferMemoryRequirements(dev, b, &req);
	VkDeviceMemory m = alloc(req, 0);
	CHECK(vkBindBufferMemory(dev, b, m, 0));
	CHECK(vkMapMemory(dev, m, 0, VK_WHOLE_SIZE, 0, &p));
	memcpy(p, data, size);
	return b;
}

static VkCommandBuffer begin(void)
{
	VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
	VkCommandBuffer cmd;
	CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
	VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
	CHECK(vkBeginCommandBuffer(cmd, &bi));
	return cmd;
}

static VkResult submit(VkCommandBuffer cmd)
{
	VkResult r = vkEndCommandBuffer(cmd);
	if (r != VK_SUCCESS)
		return r;
	VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	VkFence fence;
	CHECK(vkCreateFence(dev, &fci, NULL, &fence));
	VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd};
	r = vkQueueSubmit(queue, 1, &si, fence);
	if (r == VK_SUCCESS)
		r = vkWaitForFences(dev, 1, &fence, VK_TRUE, 20ull * 1000 * 1000 * 1000);
	return r;
}

/* A sampled texture filled by one buffer -> image copy per level (data: levels in order, each
 * layers/slices x rows x elements, tight). Padded allocation (see the header). */
struct tex { VkImage image; VkImageView view; };
static VkResult make_tex(struct tex *t, VkImageType type, VkImageViewType vt, VkFormat f, VkImageTiling tiling,
			 VkImageAspectFlags aspect, unsigned bpe, unsigned block, uint32_t w, uint32_t h, uint32_t d,
			 uint32_t levels, uint32_t layers, const void *data)
{
	VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = type, .format = f,
		.extent = {w, h, d}, .mipLevels = levels, .arrayLayers = layers, .samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = tiling, .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		.flags = vt == VK_IMAGE_VIEW_TYPE_CUBE ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0};
	CHECK(vkCreateImage(dev, &ci, NULL, &t->image));
	VkMemoryRequirements req;
	vkGetImageMemoryRequirements(dev, t->image, &req);
	VkDeviceMemory m = alloc(req, 3 * req.size + 256 * 1024);
	CHECK(vkBindImageMemory(dev, t->image, m, 0));
	VkImageViewCreateInfo vci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = t->image,
		.viewType = vt, .format = f, .subresourceRange = {aspect, 0, levels, 0, layers}};
	CHECK(vkCreateImageView(dev, &vci, NULL, &t->view));
	VkBufferImageCopy reg[16];
	size_t off = 0;
	for (uint32_t l = 0; l < levels; l++) {
		uint32_t lw = w >> l ? w >> l : 1, lh = h >> l ? h >> l : 1, ld = d >> l ? d >> l : 1;
		reg[l] = (VkBufferImageCopy){.bufferOffset = off, .imageSubresource = {aspect, l, 0, layers},
			.imageExtent = {lw, lh, ld}};
		off += (size_t)((lw + block - 1) / block) * ((lh + block - 1) / block) * ld * layers * bpe;
	}
	VkBuffer src = make_buffer(data, off, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
	VkCommandBuffer cmd = begin();
	vkCmdCopyBufferToImage(cmd, src, t->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, levels, reg);
	return submit(cmd);
}

static VkShaderModule module(const uint32_t *code, size_t bytes)
{
	VkShaderModule m;
	VkShaderModuleCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = bytes,
		.pCode = code};
	CHECK(vkCreateShaderModule(dev, &ci, NULL, &m));
	return m;
}

struct draw {
	const uint32_t *vs, *fs;
	size_t vs_bytes, fs_bytes;
	unsigned nattr, stride;
	VkVertexInputAttributeDescription attr[2];
	const void *verts;
	size_t vbytes;
	unsigned count;
	VkPrimitiveTopology topology;
	VkImageView texture; /* binding 0 (FS) */
	VkSampler sampler;
	float clear[4];
};

/* One render job into a 64x64 BGRA8 target; `out` = the target bytes (tight rows). */
static VkResult run(const struct draw *d, uint8_t *out)
{
	VkImageCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
		.format = VK_FORMAT_B8G8R8A8_UNORM, .extent = {N, N, 1}, .mipLevels = 1, .arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_LINEAR,
		.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, .initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED};
	VkImage rt;
	CHECK(vkCreateImage(dev, &ici, NULL, &rt));
	VkMemoryRequirements req;
	vkGetImageMemoryRequirements(dev, rt, &req);
	VkDeviceMemory rtm = alloc(req, 0);
	CHECK(vkBindImageMemory(dev, rt, rtm, 0));
	uint8_t *map;
	CHECK(vkMapMemory(dev, rtm, 0, VK_WHOLE_SIZE, 0, (void **)&map));
	memset(map, 0x5a, req.size);
	VkSubresourceLayout sl;
	vkGetImageSubresourceLayout(dev, rt, &(VkImageSubresource){VK_IMAGE_ASPECT_COLOR_BIT, 0, 0}, &sl);
	VkImageView rtv;
	VkImageViewCreateInfo vci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = rt,
		.viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_B8G8R8A8_UNORM,
		.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
	CHECK(vkCreateImageView(dev, &vci, NULL, &rtv));
	VkAttachmentDescription att = {.format = VK_FORMAT_B8G8R8A8_UNORM, .samples = VK_SAMPLE_COUNT_1_BIT,
		.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, .finalLayout = VK_IMAGE_LAYOUT_GENERAL};
	VkAttachmentReference ref = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
	VkSubpassDescription sub = {.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS, .colorAttachmentCount = 1,
		.pColorAttachments = &ref};
	VkRenderPassCreateInfo rpci = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, .attachmentCount = 1,
		.pAttachments = &att, .subpassCount = 1, .pSubpasses = &sub};
	VkRenderPass rp;
	CHECK(vkCreateRenderPass(dev, &rpci, NULL, &rp));
	VkFramebufferCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = rp,
		.attachmentCount = 1, .pAttachments = &rtv, .width = N, .height = N, .layers = 1};
	VkFramebuffer fb;
	CHECK(vkCreateFramebuffer(dev, &fci, NULL, &fb));
	VkDescriptorSetLayoutBinding b = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
	VkDescriptorSetLayoutCreateInfo lci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = d->texture ? 1 : 0, .pBindings = &b};
	VkDescriptorSetLayout dsl;
	CHECK(vkCreateDescriptorSetLayout(dev, &lci, NULL, &dsl));
	VkPipelineLayoutCreateInfo plci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = d->texture ? 1 : 0, .pSetLayouts = &dsl};
	VkPipelineLayout pl;
	CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
	VkDescriptorSet set = VK_NULL_HANDLE;
	if (d->texture) {
		VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
		VkDescriptorPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1,
			.poolSizeCount = 1, .pPoolSizes = &ps};
		VkDescriptorPool dp;
		CHECK(vkCreateDescriptorPool(dev, &pci, NULL, &dp));
		VkDescriptorSetAllocateInfo dai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl};
		CHECK(vkAllocateDescriptorSets(dev, &dai, &set));
		VkDescriptorImageInfo ii = {d->sampler, d->texture, VK_IMAGE_LAYOUT_GENERAL};
		VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 0,
			.descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii};
		vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);
	}
	VkPipelineShaderStageCreateInfo st[2] = {
		{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
		 .module = module(d->vs, d->vs_bytes), .pName = "main"},
		{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
		 .module = module(d->fs, d->fs_bytes), .pName = "main"}};
	VkVertexInputBindingDescription vb = {0, d->stride, VK_VERTEX_INPUT_RATE_VERTEX};
	VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &vb,
		.vertexAttributeDescriptionCount = d->nattr, .pVertexAttributeDescriptions = d->attr};
	VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = d->topology};
	VkViewport vp = {0, 0, N, N, 0, 1};
	VkRect2D sc = {{0, 0}, {N, N}};
	VkPipelineViewportStateCreateInfo vps = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1, .pViewports = &vp, .scissorCount = 1, .pScissors = &sc};
	VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1};
	VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
	VkPipelineColorBlendAttachmentState cba = {.colorWriteMask = 0xf};
	VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1, .pAttachments = &cba};
	VkGraphicsPipelineCreateInfo gci = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = 2,
		.pStages = st, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vps,
		.pRasterizationState = &rs, .pMultisampleState = &ms, .pColorBlendState = &cb, .layout = pl,
		.renderPass = rp};
	VkPipeline pipe;
	VkResult r = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gci, NULL, &pipe);
	if (r != VK_SUCCESS)
		return r;
	VkBuffer vbuf = make_buffer(d->verts, d->vbytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
	VkCommandBuffer cmd = begin();
	VkClearValue cv;
	memcpy(cv.color.float32, d->clear, 16);
	VkRenderPassBeginInfo rbi = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = rp,
		.framebuffer = fb, .renderArea = sc, .clearValueCount = 1, .pClearValues = &cv};
	vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
	if (set)
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &set, 0, NULL);
	VkDeviceSize zero = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &vbuf, &zero);
	vkCmdDraw(cmd, d->count, 1, 0, 0);
	vkCmdEndRenderPass(cmd);
	r = submit(cmd);
	if (r != VK_SUCCESS)
		return r;
	for (unsigned y = 0; y < N; y++)
		memcpy(out + y * N * 4, map + y * sl.rowPitch, N * 4);
	return VK_SUCCESS;
}

/* ---- cases ---- */
static const float quad_uv[16] = {-1, -1, 0, 0, 1, -1, 1, 0, -1, 1, 0, 1, 1, 1, 1, 1}; /* strip: pos, uv */
static const float red_tri[12] = {-0.75f, -0.75f, 0, 1, 0.75f, -0.75f, 0, 1, 0, 0.75f, 0, 1};

static VkSampler sampler(bool compare)
{
	VkSampler s;
	VkSamplerCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST,
		.minFilter = VK_FILTER_NEAREST, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = VK_LOD_CLAMP_NONE,
		.compareEnable = compare, .compareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK};
	CHECK(vkCreateSampler(dev, &ci, NULL, &s));
	return s;
}

static struct draw tex_draw(const uint32_t *fs, size_t fs_bytes, VkImageView view, VkSampler s)
{
	return (struct draw){.vs = uv_vert, .vs_bytes = sizeof(uv_vert), .fs = fs, .fs_bytes = fs_bytes, .nattr = 2,
		.stride = 16, .attr = {{0, 0, VK_FORMAT_R32G32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32_SFLOAT, 8}},
		.verts = quad_uv, .vbytes = sizeof(quad_uv), .count = 4, .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
		.texture = view, .sampler = s, .clear = {0, 0, 1, 1}};
}

/* Compare a BGRA8 readback with expected RGBA bytes; slack per pixel. */
static int compare(const char *name, const uint8_t *got, uint8_t (*want)[4], unsigned slack)
{
	unsigned bad = 0, maxd = 0;
	for (unsigned i = 0; i < N * N; i++) {
		if (skip[i])
			continue;
		const uint8_t w[4] = {want[i][2], want[i][1], want[i][0], want[i][3]}; /* B G R A */
		for (unsigned k = 0; k < 4; k++) {
			unsigned dd = (unsigned)abs((int)got[4 * i + k] - (int)w[k]);
			maxd = dd > maxd ? dd : maxd;
			if (dd > slack) {
				if (bad < 3)
					fprintf(stderr, "  %s px(%u,%u) ch%u got %02x want %02x\n", name, i % N, i / N, k,
						got[4 * i + k], w[k]);
				bad++;
				break;
			}
		}
	}
	printf("case=%s result=%s bad=%u maxdiff=%u\n", name, bad ? "BAD" : maxd ? "CLOSE" : "EXACT", bad, maxd);
	return bad ? 1 : 0;
}

static void save(const char *name, const uint8_t *px)
{
	if (!outdir)
		return;
	char path[4096];
	snprintf(path, sizeof(path), "%s/%s.raw", outdir, name);
	FILE *f = fopen(path, "wb");
	if (f) {
		fwrite(px, 4, N * N, f);
		fclose(f);
	}
}

static bool inside_tri(const float t[3][2], float x, float y)
{
	float a = 0;
	for (unsigned i = 0; i < 3; i++) {
		const float *p = t[i], *q = t[(i + 1) % 3];
		float e = (q[0] - p[0]) * (y - p[1]) - (q[1] - p[1]) * (x - p[0]);
		if (e == 0)
			return false;
		if (a == 0)
			a = e;
		else if ((e > 0) != (a > 0))
			return false;
	}
	return true;
}

/* A mip chain of `levels` (w >> L) x (h >> L) RGBA8 levels, `layers` each: (L*40 + 20 + layer*100, x, y, 255). */
static uint32_t *chain(uint32_t w, uint32_t h, unsigned levels, unsigned layers)
{
	size_t n = 0;
	for (unsigned l = 0; l < levels; l++)
		n += (size_t)(w >> l ? w >> l : 1) * (h >> l ? h >> l : 1) * layers;
	uint32_t *p = malloc(4 * n), *q = p;
	for (unsigned l = 0; l < levels; l++)
		for (unsigned a = 0; a < layers; a++)
			for (unsigned y = 0; y < (h >> l ? h >> l : 1); y++)
				for (unsigned x = 0; x < (w >> l ? w >> l : 1); x++)
					*q++ = ((l * 40 + 20 + a * 100) & 0xff) | (x & 0xff) << 8 | (y & 0xff) << 16 | 0xffu << 24;
	return p;
}

static void chain_px(uint8_t out[4], unsigned level, unsigned layer, unsigned x, unsigned y)
{
	out[0] = (uint8_t)(level * 40 + 20 + layer * 100), out[1] = (uint8_t)x, out[2] = (uint8_t)y, out[3] = 255;
}

/* uv of pixel i's centre */
static float U(unsigned i) { return ((i % N) + 0.5f) / N; }
static float V(unsigned i) { return ((i / N) + 0.5f) / N; }
static unsigned clampi(int v, int hi) { return v < 0 ? 0 : v > hi ? (unsigned)hi : (unsigned)v; }

static VkPhysicalDevice phys;

static int run_case(const char *name)
{
	static uint8_t px[N * N * 4];
	static uint8_t want[N * N][4];
	struct draw d;
	VkResult r = VK_SUCCESS;
	unsigned slack = 0;
	struct tex t;
	memset(skip, 0, sizeof(skip));
	VkSampler sn = sampler(false);
	if (!strcmp(name, "one")) {
		d = (struct draw){.vs = pass_vert, .vs_bytes = sizeof(pass_vert), .fs = solid_frag,
			.fs_bytes = sizeof(solid_frag), .nattr = 1, .stride = 16,
			.attr = {{0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0}}, .verts = red_tri, .vbytes = sizeof(red_tri),
			.count = 3, .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, .clear = {0, 0, 1, 1}};
		const float tri[3][2] = {{8, 8}, {56, 8}, {32, 56}};
		for (unsigned i = 0; i < N * N; i++) {
			bool in = inside_tri(tri, (i % N) + 0.5f, (i / N) + 0.5f);
			const uint8_t c[4] = {in ? 255 : 0, 0, in ? 0 : 255, 255};
			memcpy(want[i], c, 4);
		}
	} else if (!strcmp(name, "arr") || !strcmp(name, "cube") || !strcmp(name, "arr-layer")) {
		/* the iOS tex-array/tex-cube fill: texel (x, y) of layer L = x*60 | y*60 << 8 | L*40 << 16 | 0xff << 24 */
		static uint32_t lay[6][16];
		for (unsigned l = 0; l < 6; l++)
			for (unsigned i = 0; i < 16; i++)
				lay[l][i] = (i % 4) * 60u | ((i / 4) * 60u) << 8 | (l * 40u) << 16 | 0xffu << 24;
		bool cube = !strcmp(name, "cube");
		r = make_tex(&t, VK_IMAGE_TYPE_2D, cube ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D_ARRAY,
			     VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, 4, 1, 4, 4,
			     1, 1, cube ? 6 : 2, lay);
		d = cube ? tex_draw(SHADER(tt_cube_frag), t.view, sn) :
		    !strcmp(name, "arr") ? tex_draw(SHADER(tt_arr_frag), t.view, sn) :
					   tex_draw(SHADER(tt_arr_lod_frag), t.view, sn);
		for (unsigned i = 0; i < N * N; i++) {
			unsigned x = (unsigned)(U(i) * 4), y = (unsigned)(V(i) * 4), l = 1;
			if (cube) /* -Y: s = u, t = 1 - v (Vulkan cube face selection) */
				y = clampi((int)floorf((1 - V(i)) * 4), 3), l = 3;
			if (!strcmp(name, "arr-layer"))
				l = U(i) * 2 < 0.5f ? 0 : 1;
			const uint8_t c[4] = {(uint8_t)(x * 60), (uint8_t)(y * 60), (uint8_t)(l * 40), 255};
			memcpy(want[i], c, 4);
		}
	} else if (!strcmp(name, "npot") || !strcmp(name, "npot-lod")) {
		/* 12x10, 4 levels (A9: each level twiddled over its own extent padded to powers of two) */
		uint32_t *c = chain(12, 10, 4, 1);
		r = make_tex(&t, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL,
			     VK_IMAGE_ASPECT_COLOR_BIT, 4, 1, 12, 10, 1, 4, 1, c);
		free(c);
		bool lod = !strcmp(name, "npot-lod");
		d = lod ? tex_draw(SHADER(tt_lod_frag), t.view, sn) : tex_draw(SHADER(tex_frag), t.view, sn);
		for (unsigned i = 0; i < N * N; i++) /* level 0 magnified, or level 2 = 3x2 */
			chain_px(want[i], lod ? 2 : 0, 0, (unsigned)(U(i) * (lod ? 3 : 12)), (unsigned)(V(i) * (lod ? 2 : 10)));
	} else if (!strcmp(name, "arr-mip")) {
		uint32_t *c = chain(8, 8, 4, 2);
		r = make_tex(&t, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_FORMAT_R8G8B8A8_UNORM,
			     VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, 4, 1, 8, 8, 1, 4, 2, c);
		free(c);
		d = tex_draw(SHADER(tt_arr_lod1_frag), t.view, sn);
		for (unsigned i = 0; i < N * N; i++)
			chain_px(want[i], 1, 1, (unsigned)(U(i) * 4), (unsigned)(V(i) * 4));
	} else {
		fprintf(stderr, "tt-vk: unknown case %s\n", name);
		return 3;
	}
	if (r == VK_SUCCESS)
		r = run(&d, px);
	if (r != VK_SUCCESS) {
		printf("case=%s result=REFUSED vkresult=%d\n", name, r);
		return 4;
	}
	if (offline) {
		printf("case=%s result=ACCEPTED\n", name);
		return 0;
	}
	save(name, px);
	return compare(name, px, want, slack);
}

int main(int argc, char **argv)
{
	const char *list = "one,arr,arr-layer,cube,npot,npot-lod,arr-mip";
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--offline"))
			offline = true;
		else if (!strcmp(argv[i], "--out") && i + 1 < argc)
			outdir = argv[++i];
		else if (!strcmp(argv[i], "--case") && i + 1 < argc)
			list = argv[++i];
		else {
			fprintf(stderr, "usage: %s [--offline] [--out DIR] [--case NAME[,NAME...]]\n", argv[0]);
			return 3;
		}
	}
	VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_0};
	VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
	VkInstance inst;
	CHECK(vkCreateInstance(&ici, NULL, &inst));
	uint32_t n = 1;
	VkPhysicalDevice pd;
	VkResult er = vkEnumeratePhysicalDevices(inst, &n, &pd);
	if ((er != VK_SUCCESS && er != VK_INCOMPLETE) || !n) {
		fprintf(stderr, "tt-vk: no device\n");
		return 3;
	}
	vkGetPhysicalDeviceMemoryProperties(pd, &memprops);
	phys = pd;
	VkPhysicalDeviceFeatures have, feat = {0};
	vkGetPhysicalDeviceFeatures(pd, &have);
	feat.textureCompressionETC2 = have.textureCompressionETC2;
	float prio = 1;
	VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1,
		.pQueuePriorities = &prio};
	VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
		.pQueueCreateInfos = &qci, .pEnabledFeatures = &feat};
	CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
	vkGetDeviceQueue(dev, 0, 0, &queue);
	VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
	CHECK(vkCreateCommandPool(dev, &pci, NULL, &pool));
	int worst = 0;
	char *cases = strdup(list);
	for (char *c = strtok(cases, ","); c; c = strtok(NULL, ",")) {
		int rc = run_case(c);
		worst = rc > worst ? rc : worst;
		if (rc >= 3)
			break; /* a refused or failed job: stop the run here */
	}
	printf("result=%s\n", worst == 0 ? "PASS" : worst == 1 ? "MISMATCH" : worst == 4 ? "REFUSED" : "ERROR");
	return worst;
}
