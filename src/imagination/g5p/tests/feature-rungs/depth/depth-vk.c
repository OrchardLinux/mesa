/* SPDX-License-Identifier: MIT */
/* depth-vk: depth/stencil attachments, plain Vulkan 1.0, checked through the colour target against a CPU model.
 * Ported from the Apple A9 (G5P) Linux port's fixture, where all 18 jobs below are exact on hardware.
 * Depth is never read back or sampled: only its effect on the colour target is checked.
 *
 * Every case is a sequence of render jobs (one render pass, one submit + fence wait each) on one 64x64 linear
 * B8G8R8A8 target; the colour is checked pixel for pixel against a CPU model after every job. Every draw is a
 * pixel-aligned rectangle at one depth, so the model is exact. Cases:
 *   one         colour only (the control path through this fixture), 3 jobs
 *   depth       D32F, one job: green quad z=.75, red rect z=.5, blue quad z=.6 Less: red inside, blue outside
 *   reload      D32F, 3 jobs: red rect z=.5 (stored); LOAD + green quad z=.6 Less (stored); LOAD + blue quad
 *               z=.55 Less: red inside, blue outside (pass 2 reads pass 1's depth, pass 3 reads pass 2's)
 *   reload16    the same with D16
 *   stencil     D32F_S8, 3 jobs: red rect writes stencil 1 (Replace); LOAD + green quad Equal 1; LOAD + blue quad
 *               NotEqual 1: green inside, blue outside
 *   bias        D32F, one job: red quad z=.5, then the same quad green with a -64 constant depth bias, Less: green
 *               (without the bias Less fails and it stays red)
 *   size        reload at 100x75: the 32-pixel alignment and a non-power-of-two surface
 *   cube16      vkcube's depth: D16 cleared, LessEqual + write, store DONT_CARE (depth stays on chip): green quad
 *               z=.75, red rect z=.5, blue quad z=.6: red inside, blue outside
 *   clip        a stencil clip in one job (no LOAD), D32F_S8: red rect writes stencil 1 (Replace), green quad Equal 1,
 *               blue quad NotEqual 1: green inside, blue outside
 *   size1       100x75 in one job (no LOAD), D32F: red rect z=.5 Less, blue quad z=.6 Less: red inside, blue outside
 * Each depth/stencil image is bound 16 KiB into its own allocation with 16 KiB of 0xa5 guard bytes before and
 * after it, and the image itself starts as 0xa5. After each job: both guards (guard=, must stay 0) and the
 * footprint= the depth/stencil store wrote (the highest changed byte of the image, rounded up to 4 KiB;
 * informational: footprint <= size says the store stayed inside the image; 0 with storeOp DONT_CARE).
 *
 *   depth-vk [--offline] [--out DIR] [--case NAME[,NAME...]] [--expect-no-depth]
 *
 * Prints one line per job and result=. Exit 0 all exact, 1 a pixel mismatch, 2 a guard byte changed, 3 setup
 * error, 4 a job was refused or failed (-8 refused, -4 device lost, 2 fence timeout). --offline (a driver that
 * executes nothing, no pixels): every job must be accepted. --expect-no-depth: the device must advertise no
 * depth/stencil format; depth cases are skipped and the colour control still runs.
 */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#include "color.frag.h"
#include "color.vert.h"

#define CHECK(x)                                                                            \
	do {                                                                                \
		VkResult r_ = (x);                                                          \
		if (r_ != VK_SUCCESS) {                                                     \
			fprintf(stderr, "depth-vk: %s = %d (line %d)\n", #x, r_, __LINE__); \
			exit(3);                                                            \
		}                                                                           \
	} while (0)

#define GUARD 16384u
#define MAXJOBS 3
#define MAXDRAWS 3

static VkDevice dev;
static VkPhysicalDevice phys;
static VkQueue queue;
static VkCommandPool pool;
static VkPhysicalDeviceMemoryProperties memprops;
static bool offline;
static const char *outdir;

enum { RED = 0xffff0000u, GREEN = 0xff00ff00u, BLUE = 0xff0000ffu, BLACK = 0xff000000u };

struct draw {
	int x0, y0, x1, y1; /* pixels; 0,0,W,H = full */
	float z;
	uint32_t color;
	VkCompareOp op;      /* depth compare (ALWAYS + no write: test off) */
	bool write;
	int stencil;         /* 0 none, 1 write ref 1 (Replace, Always), 2 Equal 1, 3 NotEqual 1 */
	float bias;          /* constant depth bias factor */
};

struct job {
	bool load;           /* colour and depth/stencil LOAD (else CLEAR black / 1.0 / 0) */
	unsigned ndraws;
	struct draw d[MAXDRAWS];
	uint32_t want_in, want_out; /* expected colour inside / outside the rect (0x0: not checked) */
};

struct tcase {
	const char *name;
	VkFormat ds;
	uint32_t w, h;
	unsigned njobs;
	struct job j[MAXJOBS];
	int rx0, ry0, rx1, ry1;   /* the "inside" rectangle of the checks */
	bool no_store;            /* depth/stencil storeOp DONT_CARE (vkcube) */
};

#define FULL(z_, c_, op_, wr_) {0, 0, 0, 0, z_, c_, op_, wr_, 0, 0}
#define RECT(z_, c_, op_, wr_) {16, 24, 48, 56, z_, c_, op_, wr_, 0, 0}

static const struct tcase cases[] = {
	{"one", VK_FORMAT_UNDEFINED, 64, 64, 3, {
		{false, 1, {RECT(.5f, RED, VK_COMPARE_OP_ALWAYS, false)}, RED, BLACK},
		{true, 1, {FULL(.5f, GREEN, VK_COMPARE_OP_ALWAYS, false)}, GREEN, GREEN},
		{false, 1, {RECT(.5f, BLUE, VK_COMPARE_OP_ALWAYS, false)}, BLUE, BLACK}}, 16, 24, 48, 56, false},
	{"depth", VK_FORMAT_D32_SFLOAT, 64, 64, 1, {
		{false, 3, {FULL(.75f, GREEN, VK_COMPARE_OP_LESS, true), RECT(.5f, RED, VK_COMPARE_OP_LESS, true),
			    FULL(.6f, BLUE, VK_COMPARE_OP_LESS, true)}, RED, BLUE}}, 16, 24, 48, 56, false},
	{"reload", VK_FORMAT_D32_SFLOAT, 64, 64, 3, {
		{false, 1, {RECT(.5f, RED, VK_COMPARE_OP_LESS, true)}, RED, BLACK},
		{true, 1, {FULL(.6f, GREEN, VK_COMPARE_OP_LESS, true)}, RED, GREEN},
		{true, 1, {FULL(.55f, BLUE, VK_COMPARE_OP_LESS, true)}, RED, BLUE}}, 16, 24, 48, 56, false},
	{"reload16", VK_FORMAT_D16_UNORM, 64, 64, 3, {
		{false, 1, {RECT(.5f, RED, VK_COMPARE_OP_LESS, true)}, RED, BLACK},
		{true, 1, {FULL(.6f, GREEN, VK_COMPARE_OP_LESS, true)}, RED, GREEN},
		{true, 1, {FULL(.55f, BLUE, VK_COMPARE_OP_LESS, true)}, RED, BLUE}}, 16, 24, 48, 56, false},
	{"stencil", VK_FORMAT_D32_SFLOAT_S8_UINT, 64, 64, 3, {
		{false, 1, {{16, 24, 48, 56, .5f, RED, VK_COMPARE_OP_ALWAYS, false, 1, 0}}, RED, BLACK},
		{true, 1, {{0, 0, 0, 0, .5f, GREEN, VK_COMPARE_OP_ALWAYS, false, 2, 0}}, GREEN, BLACK},
		{true, 1, {{0, 0, 0, 0, .5f, BLUE, VK_COMPARE_OP_ALWAYS, false, 3, 0}}, GREEN, BLUE}}, 16, 24, 48, 56, false},
	{"bias", VK_FORMAT_D32_SFLOAT, 64, 64, 1, {
		{false, 2, {FULL(.5f, RED, VK_COMPARE_OP_LESS, true),
			    {0, 0, 0, 0, .5f, GREEN, VK_COMPARE_OP_LESS, true, 0, -64.f}}, GREEN, GREEN}}, 16, 24, 48, 56, false},
	{"size", VK_FORMAT_D32_SFLOAT, 100, 75, 3, {
		{false, 1, {{20, 10, 70, 60, .5f, RED, VK_COMPARE_OP_LESS, true, 0, 0}}, RED, BLACK},
		{true, 1, {FULL(.6f, GREEN, VK_COMPARE_OP_LESS, true)}, RED, GREEN},
		{true, 1, {FULL(.55f, BLUE, VK_COMPARE_OP_LESS, true)}, RED, BLUE}}, 20, 10, 70, 60, false},
	{"cube16", VK_FORMAT_D16_UNORM, 64, 64, 1, {
		{false, 3, {FULL(.75f, GREEN, VK_COMPARE_OP_LESS_OR_EQUAL, true), RECT(.5f, RED, VK_COMPARE_OP_LESS_OR_EQUAL, true),
			    FULL(.6f, BLUE, VK_COMPARE_OP_LESS_OR_EQUAL, true)}, RED, BLUE}}, 16, 24, 48, 56, true},
	{"clip", VK_FORMAT_D32_SFLOAT_S8_UINT, 64, 64, 1, {
		{false, 3, {{16, 24, 48, 56, .5f, RED, VK_COMPARE_OP_ALWAYS, false, 1, 0},
			    {0, 0, 0, 0, .5f, GREEN, VK_COMPARE_OP_ALWAYS, false, 2, 0},
			    {0, 0, 0, 0, .5f, BLUE, VK_COMPARE_OP_ALWAYS, false, 3, 0}}, GREEN, BLUE}}, 16, 24, 48, 56, false},
	{"size1", VK_FORMAT_D32_SFLOAT, 100, 75, 1, {
		{false, 2, {{20, 10, 70, 60, .5f, RED, VK_COMPARE_OP_LESS, true, 0, 0},
			    FULL(.6f, BLUE, VK_COMPARE_OP_LESS, true)}, RED, BLUE}}, 20, 10, 70, 60, false},
};

static uint32_t mem_type(uint32_t bits)
{
	for (uint32_t i = 0; i < memprops.memoryTypeCount; i++)
		if ((bits & (1u << i)) && (memprops.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
			return i;
	fprintf(stderr, "depth-vk: no host-visible memory type\n");
	exit(3);
}

struct surface {
	VkImage image;
	VkImageView view;
	VkDeviceMemory mem;
	uint8_t *map;      /* the whole allocation: guard, image, guard */
	VkDeviceSize size; /* the image's */
	uint32_t pitch;
};

static void make_surface(struct surface *s, VkFormat f, uint32_t w, uint32_t h, bool color)
{
	VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
		.format = f, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = color ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL,
		.usage = color ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
	CHECK(vkCreateImage(dev, &ci, NULL, &s->image));
	VkMemoryRequirements req;
	vkGetImageMemoryRequirements(dev, s->image, &req);
	if (GUARD % req.alignment) {
		fprintf(stderr, "depth-vk: image alignment %llu does not divide the guard\n", (unsigned long long)req.alignment);
		exit(3);
	}
	s->size = req.size;
	VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = GUARD + req.size + GUARD, .memoryTypeIndex = mem_type(req.memoryTypeBits)};
	CHECK(vkAllocateMemory(dev, &ai, NULL, &s->mem));
	CHECK(vkBindImageMemory(dev, s->image, s->mem, GUARD));
	CHECK(vkMapMemory(dev, s->mem, 0, VK_WHOLE_SIZE, 0, (void **)&s->map));
	memset(s->map, 0xa5, ai.allocationSize);
	s->pitch = w * 4;
	if (color) {
		VkSubresourceLayout l;
		vkGetImageSubresourceLayout(dev, s->image, &(VkImageSubresource){VK_IMAGE_ASPECT_COLOR_BIT, 0, 0}, &l);
		s->pitch = (uint32_t)l.rowPitch;
	}
	VkImageAspectFlags aspects = color ? VK_IMAGE_ASPECT_COLOR_BIT :
		VK_IMAGE_ASPECT_DEPTH_BIT | (f == VK_FORMAT_D32_SFLOAT_S8_UINT ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
	VkImageViewCreateInfo vci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = s->image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D, .format = f, .subresourceRange = {aspects, 0, 1, 0, 1}};
	CHECK(vkCreateImageView(dev, &vci, NULL, &s->view));
}

static uint64_t guard_bad(const struct surface *s)
{
	uint64_t bad = 0;
	for (uint32_t i = 0; i < GUARD; i++)
		bad += (s->map[i] != 0xa5) + (s->map[GUARD + s->size + i] != 0xa5);
	return bad;
}

/* the ZLS footprint: the highest byte of the image that is no longer 0xa5, rounded up to 4 KiB */
static uint64_t footprint(const struct surface *s)
{
	for (uint64_t i = s->size; i > 0; i--)
		if (s->map[GUARD + i - 1] != 0xa5)
			return (i + 4095) & ~4095ull;
	return 0;
}

static VkRenderPass make_pass(VkFormat ds, bool load, bool no_store)
{
	VkAttachmentLoadOp lo = load ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
	VkAttachmentDescription att[2] = {
		{.format = VK_FORMAT_B8G8R8A8_UNORM, .samples = VK_SAMPLE_COUNT_1_BIT, .loadOp = lo,
		 .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
		 .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
		 .initialLayout = load ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
		 .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
		{.format = ds, .samples = VK_SAMPLE_COUNT_1_BIT, .loadOp = lo,
		 .storeOp = no_store ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE,
		 .stencilLoadOp = no_store ? VK_ATTACHMENT_LOAD_OP_DONT_CARE : lo,
		 .stencilStoreOp = no_store ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE,
		 .initialLayout = load ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
		 .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL}};
	VkAttachmentReference cref = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
	VkAttachmentReference dref = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
	VkSubpassDescription sub = {.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS, .colorAttachmentCount = 1,
		.pColorAttachments = &cref, .pDepthStencilAttachment = ds ? &dref : NULL};
	VkRenderPassCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, .attachmentCount = ds ? 2 : 1,
		.pAttachments = att, .subpassCount = 1, .pSubpasses = &sub};
	VkRenderPass rp;
	CHECK(vkCreateRenderPass(dev, &ci, NULL, &rp));
	return rp;
}

static VkResult make_pipeline(const struct draw *d, VkRenderPass rp, VkPipelineLayout layout, VkPipeline *out)
{
	VkShaderModuleCreateInfo vsi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = sizeof(spv_color_vert), .pCode = spv_color_vert};
	VkShaderModuleCreateInfo fsi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = sizeof(spv_color_frag), .pCode = spv_color_frag};
	VkShaderModule vs, fs;
	CHECK(vkCreateShaderModule(dev, &vsi, NULL, &vs));
	CHECK(vkCreateShaderModule(dev, &fsi, NULL, &fs));
	VkPipelineShaderStageCreateInfo st[2] = {
		{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
		 .module = vs, .pName = "main"},
		{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
		 .module = fs, .pName = "main"}};
	VkVertexInputBindingDescription b = {.stride = 32};
	VkVertexInputAttributeDescription a[2] = {{.location = 0, .format = VK_FORMAT_R32G32B32A32_SFLOAT},
		{.location = 1, .format = VK_FORMAT_R32G32B32A32_SFLOAT, .offset = 16}};
	VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &b,
		.vertexAttributeDescriptionCount = 2, .pVertexAttributeDescriptions = a};
	VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
	VkPipelineViewportStateCreateInfo vp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1, .scissorCount = 1};
	VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL, .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1,
		.depthBiasEnable = d->bias != 0, .depthBiasConstantFactor = d->bias,
		.depthBiasClamp = d->bias ? -.01f : 0}; /* clamps nothing whether the clamp is signed or a magnitude */
	VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
	static const VkCompareOp sops[4] = {VK_COMPARE_OP_ALWAYS, VK_COMPARE_OP_ALWAYS, VK_COMPARE_OP_EQUAL,
					    VK_COMPARE_OP_NOT_EQUAL};
	VkStencilOpState so = {.failOp = VK_STENCIL_OP_KEEP, .depthFailOp = VK_STENCIL_OP_KEEP,
		.passOp = d->stencil == 1 ? VK_STENCIL_OP_REPLACE : VK_STENCIL_OP_KEEP, .compareOp = sops[d->stencil],
		.compareMask = 0xff, .writeMask = 0xff, .reference = 1};
	VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = d->op != VK_COMPARE_OP_ALWAYS || d->write, .depthWriteEnable = d->write,
		.depthCompareOp = d->op, .stencilTestEnable = d->stencil != 0, .front = so, .back = so};
	VkPipelineColorBlendAttachmentState cba = {.colorWriteMask = 0xf};
	VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1, .pAttachments = &cba};
	VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2, .pDynamicStates = dyn};
	VkGraphicsPipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = 2,
		.pStages = st, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vp,
		.pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds, .pColorBlendState = &cb,
		.pDynamicState = &dy, .layout = layout, .renderPass = rp};
	VkResult r = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &ci, NULL, out);
	vkDestroyShaderModule(dev, vs, NULL);
	vkDestroyShaderModule(dev, fs, NULL);
	return r;
}

static void rect_verts(float *v, const struct draw *d, uint32_t w, uint32_t h)
{
	int x0 = d->x1 ? d->x0 : 0, y0 = d->x1 ? d->y0 : 0, x1 = d->x1 ? d->x1 : (int)w, y1 = d->x1 ? d->y1 : (int)h;
	float fx0 = 2.f * x0 / w - 1, fx1 = 2.f * x1 / w - 1, fy0 = 2.f * y0 / h - 1, fy1 = 2.f * y1 / h - 1;
	const float c[4] = {((d->color >> 16) & 0xff) / 255.f, ((d->color >> 8) & 0xff) / 255.f, (d->color & 0xff) / 255.f, 1};
	const float p[6][2] = {{fx0, fy0}, {fx1, fy0}, {fx1, fy1}, {fx0, fy0}, {fx1, fy1}, {fx0, fy1}};
	for (int i = 0; i < 6; i++) {
		float *o = v + 8 * i;
		o[0] = p[i][0], o[1] = p[i][1], o[2] = d->z, o[3] = 1;
		memcpy(o + 4, c, 16);
	}
}

static double now_ms(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static bool depth_supported(VkFormat f)
{
	VkFormatProperties p;
	vkGetPhysicalDeviceFormatProperties(phys, f, &p);
	return p.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
}

static void dump(const struct surface *s, uint32_t w, uint32_t h, const char *name)
{
	if (!outdir)
		return;
	char path[512];
	snprintf(path, sizeof(path), "%s/%s", outdir, name);
	FILE *f = fopen(path, "wb");
	if (!f)
		return;
	for (uint32_t y = 0; y < h; y++)
		fwrite(s->map + GUARD + (size_t)y * s->pitch, 4, w, f);
	fclose(f);
}

static int run_case(const struct tcase *k)
{
	const uint32_t W = k->w, H = k->h;
	const bool ds = k->ds != VK_FORMAT_UNDEFINED;
	printf("case=%s format=%d size=%ux%u jobs=%u\n", k->name, k->ds, W, H, k->njobs);
	struct surface rt, z = {0};
	make_surface(&rt, VK_FORMAT_B8G8R8A8_UNORM, W, H, true);
	if (ds)
		make_surface(&z, k->ds, W, H, false);
	VkRenderPass rp[2] = {make_pass(k->ds, false, k->no_store), make_pass(k->ds, true, k->no_store)};
	VkFramebuffer fb[2];
	for (int i = 0; i < 2; i++) {
		VkImageView v[2] = {rt.view, z.view};
		VkFramebufferCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = rp[i],
			.attachmentCount = ds ? 2 : 1, .pAttachments = v, .width = W, .height = H, .layers = 1};
		CHECK(vkCreateFramebuffer(dev, &fci, NULL, &fb[i]));
	}
	VkPipelineLayoutCreateInfo lci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	VkPipelineLayout layout;
	CHECK(vkCreatePipelineLayout(dev, &lci, NULL, &layout));
	VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = MAXJOBS * MAXDRAWS * 6 * 32,
		.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT};
	VkBuffer vb;
	CHECK(vkCreateBuffer(dev, &bci, NULL, &vb));
	VkMemoryRequirements req;
	vkGetBufferMemoryRequirements(dev, vb, &req);
	VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
		.memoryTypeIndex = mem_type(req.memoryTypeBits)};
	VkDeviceMemory vmem;
	float *verts;
	CHECK(vkAllocateMemory(dev, &ai, NULL, &vmem));
	CHECK(vkBindBufferMemory(dev, vb, vmem, 0));
	CHECK(vkMapMemory(dev, vmem, 0, VK_WHOLE_SIZE, 0, (void **)&verts));
	int rc = 0;
	for (unsigned jn = 0; jn < k->njobs && !rc; jn++) {
		const struct job *jb = &k->j[jn];
		VkPipeline pipes[MAXDRAWS] = {0};
		for (unsigned i = 0; i < jb->ndraws; i++) {
			rect_verts(verts + (jn * MAXDRAWS + i) * 48, &jb->d[i], W, H);
			VkResult r = make_pipeline(&jb->d[i], rp[0], layout, &pipes[i]);
			if (r != VK_SUCCESS) {
				printf("case=%s job=%u pipeline refused (%d)\n", k->name, jn, r);
				return 4;
			}
		}
		VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
		VkCommandBuffer cmd;
		CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
		VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
		CHECK(vkBeginCommandBuffer(cmd, &bi));
		VkClearValue cv[2] = {{.color.float32 = {0, 0, 0, 1}}, {.depthStencil = {1.0f, 0}}};
		VkRenderPassBeginInfo rbi = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = rp[jb->load],
			.framebuffer = fb[jb->load], .renderArea = {{0, 0}, {W, H}}, .clearValueCount = ds ? 2 : 1,
			.pClearValues = cv};
		vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
		VkViewport v = {0, 0, (float)W, (float)H, 0, 1};
		VkRect2D sc = {{0, 0}, {W, H}};
		vkCmdSetViewport(cmd, 0, 1, &v);
		vkCmdSetScissor(cmd, 0, 1, &sc);
		for (unsigned i = 0; i < jb->ndraws; i++) {
			VkDeviceSize off = (VkDeviceSize)(jn * MAXDRAWS + i) * 6 * 32;
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes[i]);
			vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &off);
			vkCmdDraw(cmd, 6, 1, 0, 0);
		}
		vkCmdEndRenderPass(cmd);
		VkResult r = vkEndCommandBuffer(cmd);
		double t0 = now_ms();
		if (r == VK_SUCCESS) {
			VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd};
			r = vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
			if (r == VK_SUCCESS)
				r = vkQueueWaitIdle(queue);
		}
		double ms = now_ms() - t0;
		vkFreeCommandBuffers(dev, pool, 1, &cmd);
		for (unsigned i = 0; i < jb->ndraws; i++)
			vkDestroyPipeline(dev, pipes[i], NULL);
		if (r != VK_SUCCESS) {
			printf("case=%s job=%u result=FAILED vk=%d\n", k->name, jn, r);
			rc = 4;
			break;
		}
		if (offline) {
			printf("case=%s job=%u result=ACCEPTED ms=%.1f\n", k->name, jn, ms);
			continue;
		}
		uint64_t bad = 0, gb = guard_bad(&rt) + (ds ? guard_bad(&z) : 0);
		uint32_t first = 0, fx = 0, fy = 0;
		for (uint32_t y = 0; y < H; y++)
			for (uint32_t x = 0; x < W; x++) {
				bool in = (int)x >= k->rx0 && (int)x < k->rx1 && (int)y >= k->ry0 && (int)y < k->ry1;
				uint32_t want = in ? jb->want_in : jb->want_out, got;
				memcpy(&got, rt.map + GUARD + (size_t)y * rt.pitch + 4 * x, 4);
				if (got != want && !bad++)
					first = got, fx = x, fy = y;
			}
		char name[96];
		snprintf(name, sizeof(name), "rt-%s-j%u-after.raw", k->name, jn);
		dump(&rt, W, H, name);
		printf("case=%s job=%u result=%s bad=%llu guard=%llu footprint=%llu size=%llu ms=%.1f", k->name, jn,
		       bad ? "MISMATCH" : "EXACT", (unsigned long long)bad, (unsigned long long)gb,
		       (unsigned long long)(ds ? footprint(&z) : 0), (unsigned long long)z.size, ms);
		if (bad)
			printf(" first=(%u,%u) got=%08x want=%08x", fx, fy, first,
			       ((int)fx >= k->rx0 && (int)fx < k->rx1 && (int)fy >= k->ry0 && (int)fy < k->ry1) ?
			       jb->want_in : jb->want_out);
		printf("\n");
		if (gb)
			rc = 2;
		else if (bad)
			rc = 1;
	}
	return rc;
}

int main(int argc, char **argv)
{
	const char *only = NULL;
	bool expect_no_depth = false;
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--offline"))
			offline = true;
		else if (!strcmp(argv[i], "--out") && i + 1 < argc)
			outdir = argv[++i];
		else if (!strcmp(argv[i], "--case") && i + 1 < argc)
			only = argv[++i];
		else if (!strcmp(argv[i], "--expect-no-depth"))
			expect_no_depth = true;
		else {
			fprintf(stderr, "usage: %s [--offline] [--out DIR] [--case NAME[,NAME...]] [--expect-no-depth]\n", argv[0]);
			return 3;
		}
	}
	VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_0};
	VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
	VkInstance inst;
	CHECK(vkCreateInstance(&ici, NULL, &inst));
	uint32_t n = 1;
	VkResult er = vkEnumeratePhysicalDevices(inst, &n, &phys);
	if ((er != VK_SUCCESS && er != VK_INCOMPLETE) || !n)
		return fprintf(stderr, "depth-vk: no physical device (%d)\n", er), 3;
	vkGetPhysicalDeviceMemoryProperties(phys, &memprops);
	float prio = 1;
	VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1,
		.pQueuePriorities = &prio};
	VkPhysicalDeviceFeatures feat = {.depthBiasClamp = VK_TRUE};
	VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
		.pQueueCreateInfos = &qci, .pEnabledFeatures = &feat};
	CHECK(vkCreateDevice(phys, &dci, NULL, &dev));
	vkGetDeviceQueue(dev, 0, 0, &queue);
	VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
	CHECK(vkCreateCommandPool(dev, &pci, NULL, &pool));
	const VkFormat fmts[4] = {VK_FORMAT_D16_UNORM, VK_FORMAT_D32_SFLOAT, VK_FORMAT_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT};
	unsigned have = 0;
	for (int i = 0; i < 4; i++)
		have += depth_supported(fmts[i]);
	printf("depth formats=%u/4 D24S8=%d\n", have, depth_supported(VK_FORMAT_D24_UNORM_S8_UINT));
	if (expect_no_depth ? have != 0 : have != 4) {
		printf("result=FAIL depth formats %u, want %u\n", have, expect_no_depth ? 0 : 4);
		return 3;
	}
	char list[256];
	snprintf(list, sizeof(list), "%s", only ? only : "one,depth,reload,reload16,stencil,bias,size,cube16");
	int rc = 0, ran = 0;
	for (char *name = strtok(list, ","); name; name = strtok(NULL, ",")) {
		const struct tcase *k = NULL;
		for (unsigned c = 0; c < sizeof(cases) / sizeof(cases[0]); c++)
			if (!strcmp(name, cases[c].name))
				k = &cases[c];
		if (!k)
			return fprintf(stderr, "depth-vk: no case %s\n", name), 3;
		if (expect_no_depth && k->ds != VK_FORMAT_UNDEFINED) {
			printf("case=%s skipped (no depth formats)\n", k->name);
			continue;
		}
		int r = run_case(k);
		ran++;
		if (r && !rc)
			rc = r;
	}
	if (!ran)
		return fprintf(stderr, "depth-vk: no case\n"), 3;
	printf("result=%s\n", rc ? "FAIL" : (offline ? "ACCEPTED" : "EXACT"));
	return rc;
}
