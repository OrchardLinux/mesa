/* SPDX-License-Identifier: MIT */
/* mrt-vk: several colour targets in one render pass, plain Vulkan 1.0, checked against a CPU model.
 * Ported from the Apple A9 (G5P) Linux port's fixture, where every case below is exact on hardware.
 *
 * Each case is one process-wide device and THREE consecutive render jobs, each one submit + fence wait:
 *   one        1 target  (the control: the one-target path through the same fixture)
 *   mrt2       2 targets (clears C/M, constant outputs R/G; the scene of Apple's iOS mrt2 capture)
 *   mrt4       4 targets (clears C/M/W/K, outputs R/G/B/Y; Apple's mrt4 scene)
 *   mrt4-vary  4 targets whose outputs come from one varying by swizzle (register-form writes into output
 *              halves 4..15, a form no Apple program showed on the A9; runs only when named)
 * Job k clears target t with colour (t + k) mod 4 of C/M/W/K and draws a different pixel-aligned quad, so a
 * job that stored stale contents, another target's clear or another target's output is caught.
 *
 * Every target is a 64x64 linear B8G8R8A8 image in its own allocation, bound 16 KiB into it, with 16 KiB of
 * guard bytes (0xa5) before and after; the image starts as a per-target sentinel 0x5a5a00tt. After each job:
 * the exact per-pixel model (output inside the quad, the target's clear outside) and both guards of every
 * target. --out DIR keeps every job's targets as DIR/rt-<case>-j<k>-<t>-after.raw (64x64x4, tight rows).
 *
 *   mrt-vk [--offline] [--out DIR] [--case NAME[,NAME...]] [--expect-max N]
 *
 * Cases run in the order given, on one device: `--case one,mrt2` makes the one-target pass the device's first
 * job, so every several-target job is a later job. Default: one, mrt2, mrt4.
 *
 * Prints one line per job and result=. Exit 0 all exact, 1 a pixel mismatch, 2 a guard byte changed, 3 setup
 * error, 4 a job was refused or failed (the printed VkResult says which: -8 refused, -4 device lost, 2 fence
 * timeout). --offline (a driver that executes nothing: no pixels) checks only that every job is accepted.
 * --expect-max 1: the device must advertise one colour attachment, and every several-target case must be
 * refused before submit (the option-off build).
 */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#include "mrt-vary.vert.h"
#include "mrt2.frag.h"
#include "mrt4-vary.frag.h"
#include "mrt4.frag.h"
#include "one.frag.h"
#include "pass.vert.h"

#define CHECK(x)                                                                          \
	do {                                                                              \
		VkResult r_ = (x);                                                        \
		if (r_ != VK_SUCCESS) {                                                   \
			fprintf(stderr, "mrt-vk: %s = %d (line %d)\n", #x, r_, __LINE__); \
			exit(3);                                                          \
		}                                                                         \
	} while (0)

#define W 64
#define H 64
#define GUARD 16384u
#define JOBS 3

static VkDevice dev;
static VkQueue queue;
static VkCommandPool pool;
static VkPhysicalDeviceMemoryProperties memprops;
static bool offline;
static const char *outdir;

struct tcase {
	const char *name;
	unsigned targets;
	const uint32_t *vs, *fs;
	size_t vs_size, fs_size;
	bool inferred;
};

static const struct tcase cases[] = {
	{"one", 1, spv_pass_vert, spv_one_frag, sizeof(spv_pass_vert), sizeof(spv_one_frag), false},
	{"mrt2", 2, spv_pass_vert, spv_mrt2_frag, sizeof(spv_pass_vert), sizeof(spv_mrt2_frag), false},
	{"mrt4", 4, spv_pass_vert, spv_mrt4_frag, sizeof(spv_pass_vert), sizeof(spv_mrt4_frag), false},
	{"mrt4-vary", 4, spv_mrt_vary_vert, spv_mrt4_vary_frag, sizeof(spv_mrt_vary_vert), sizeof(spv_mrt4_vary_frag), true},
};

/* 0/1 channels, exact under any UNORM rounding */
static const float clear_c[4][4] = {{0, 1, 1, 1}, {1, 0, 1, 1}, {1, 1, 1, 1}, {0, 0, 0, 1}}; /* C M W K */
static const uint32_t out_w[4] = {0xffff0000, 0xff00ff00, 0xff0000ff, 0xffffff00};           /* R G B Y */
struct rect { uint32_t x0, y0, x1, y1; };
/* job 0 is the iOS capture's coverage shape (inside 32,40; outside 0,0); every quad keeps those two probes */
static const struct rect quads[JOBS] = {{16, 24, 48, 56}, {8, 8, 40, 48}, {24, 32, 64, 60}};

static uint32_t bgra(const float c[4])
{
	uint32_t q[4];
	for (int i = 0; i < 4; i++)
		q[i] = (uint32_t)lrintf((c[i] < 0 ? 0 : c[i] > 1 ? 1 : c[i]) * 255.0f);
	return q[2] | q[1] << 8 | q[0] << 16 | q[3] << 24;
}

static uint32_t mem_type(uint32_t bits)
{
	for (uint32_t i = 0; i < memprops.memoryTypeCount; i++)
		if ((bits & (1u << i)) && (memprops.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
			return i;
	fprintf(stderr, "mrt-vk: no host-visible memory type\n");
	exit(3);
}

struct target {
	VkImage img;
	VkImageView view;
	VkDeviceMemory mem;
	uint8_t *map;     /* whole allocation: guard, image, guard */
	VkDeviceSize size, row_pitch, img_offset, img_bytes;
};

static void make_target(struct target *t, unsigned i)
{
	VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
		.format = VK_FORMAT_B8G8R8A8_UNORM, .extent = {W, H, 1}, .mipLevels = 1, .arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_LINEAR,
		.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
	CHECK(vkCreateImage(dev, &ii, NULL, &t->img));
	VkMemoryRequirements mr;
	vkGetImageMemoryRequirements(dev, t->img, &mr);
	if (GUARD % mr.alignment) {
		fprintf(stderr, "mrt-vk: image alignment %llu does not divide the guard\n", (unsigned long long)mr.alignment);
		exit(3);
	}
	t->img_offset = GUARD;
	t->img_bytes = mr.size;
	t->size = GUARD + mr.size + GUARD;
	VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = t->size,
				   .memoryTypeIndex = mem_type(mr.memoryTypeBits)};
	CHECK(vkAllocateMemory(dev, &ai, NULL, &t->mem));
	CHECK(vkMapMemory(dev, t->mem, 0, VK_WHOLE_SIZE, 0, (void **)&t->map));
	memset(t->map, 0xa5, t->size);
	CHECK(vkBindImageMemory(dev, t->img, t->mem, t->img_offset));
	VkSubresourceLayout lay;
	vkGetImageSubresourceLayout(dev, t->img, &(VkImageSubresource){VK_IMAGE_ASPECT_COLOR_BIT, 0, 0}, &lay);
	t->row_pitch = lay.rowPitch;
	t->img_offset += lay.offset;
	for (uint32_t y = 0; y < H; y++)
		for (uint32_t x = 0; x < W; x++) {
			uint32_t s = 0x5a5a0000u | i;
			memcpy(t->map + t->img_offset + y * t->row_pitch + 4 * x, &s, 4);
		}
	VkImageViewCreateInfo vci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = t->img,
		.viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_B8G8R8A8_UNORM,
		.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
	CHECK(vkCreateImageView(dev, &vci, NULL, &t->view));
}

/* guard bytes still 0xa5: before the image, after it, and the row padding (none at 64 px) */
static uint64_t guard_bad(const struct target *t)
{
	uint64_t bad = 0;
	for (VkDeviceSize i = 0; i < t->size; i++) {
		bool in_img = i >= t->img_offset && i < t->img_offset + t->row_pitch * (H - 1) + 4 * W &&
			      (i - t->img_offset) % t->row_pitch < 4 * W;
		if (!in_img && t->map[i] != 0xa5)
			bad++;
	}
	return bad;
}

static VkRenderPass make_pass(unsigned n)
{
	VkAttachmentDescription a[4];
	VkAttachmentReference ref[4];
	for (unsigned i = 0; i < n; i++) {
		a[i] = (VkAttachmentDescription){.format = VK_FORMAT_B8G8R8A8_UNORM, .samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, .finalLayout = VK_IMAGE_LAYOUT_GENERAL};
		ref[i] = (VkAttachmentReference){i, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
	}
	VkSubpassDescription sub = {.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS, .colorAttachmentCount = n,
				    .pColorAttachments = ref};
	VkRenderPassCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, .attachmentCount = n,
				     .pAttachments = a, .subpassCount = 1, .pSubpasses = &sub};
	VkRenderPass rp;
	CHECK(vkCreateRenderPass(dev, &ci, NULL, &rp));
	return rp;
}

static VkResult make_pipeline(const struct tcase *k, VkRenderPass rp, VkPipelineLayout layout, VkPipeline *out)
{
	VkShaderModuleCreateInfo vci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = k->vs_size, .pCode = k->vs};
	VkShaderModuleCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = k->fs_size, .pCode = k->fs};
	VkShaderModule vs, fs;
	CHECK(vkCreateShaderModule(dev, &vci, NULL, &vs));
	CHECK(vkCreateShaderModule(dev, &fci, NULL, &fs));
	VkPipelineShaderStageCreateInfo st[2] = {
		{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main"},
		{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main"}};
	VkVertexInputBindingDescription bind = {.stride = 16};
	VkVertexInputAttributeDescription attr = {.location = 0, .format = VK_FORMAT_R32G32B32A32_SFLOAT};
	VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &bind, .vertexAttributeDescriptionCount = 1,
		.pVertexAttributeDescriptions = &attr};
	VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
						     .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
	VkPipelineViewportStateCreateInfo vp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
	VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1};
	VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
						   .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
	VkPipelineColorBlendAttachmentState att[4];
	for (unsigned i = 0; i < 4; i++)
		att[i] = (VkPipelineColorBlendAttachmentState){.colorWriteMask = 0xf};
	VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
						  .attachmentCount = k->targets, .pAttachments = att};
	VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	VkPipelineDynamicStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
					       .dynamicStateCount = 2, .pDynamicStates = dyn};
	VkGraphicsPipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = 2,
		.pStages = st, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vp,
		.pRasterizationState = &rs, .pMultisampleState = &ms, .pColorBlendState = &cb, .pDynamicState = &ds,
		.layout = layout, .renderPass = rp};
	VkResult r = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &ci, NULL, out);
	vkDestroyShaderModule(dev, vs, NULL);
	vkDestroyShaderModule(dev, fs, NULL);
	return r;
}

static double now_ms(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static void dump(const struct target *t, const char *name)
{
	char path[4096];
	snprintf(path, sizeof(path), "%s/%s", outdir, name);
	FILE *f = fopen(path, "wb");
	for (uint32_t y = 0; f && y < H; y++)
		fwrite(t->map + t->img_offset + y * t->row_pitch, 4, W, f);
	if (f)
		fclose(f);
}

/* 0 exact, 1 pixels, 2 guard, 4 refused/failed */
static int run_case(const struct tcase *k, unsigned max_targets)
{
	struct target t[4];
	for (unsigned i = 0; i < k->targets; i++)
		make_target(&t[i], i);
	VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = JOBS * 6 * 16,
				  .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT};
	VkBuffer vb;
	CHECK(vkCreateBuffer(dev, &bci, NULL, &vb));
	VkMemoryRequirements bmr;
	vkGetBufferMemoryRequirements(dev, vb, &bmr);
	VkDeviceMemory bmem;
	VkMemoryAllocateInfo bai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = bmr.size,
				    .memoryTypeIndex = mem_type(bmr.memoryTypeBits)};
	CHECK(vkAllocateMemory(dev, &bai, NULL, &bmem));
	float *v;
	CHECK(vkMapMemory(dev, bmem, 0, VK_WHOLE_SIZE, 0, (void **)&v));
	CHECK(vkBindBufferMemory(dev, vb, bmem, 0));
	for (unsigned j = 0; j < JOBS; j++) {
		const struct rect *q = &quads[j];
		float x0 = 2.0f * q->x0 / W - 1, x1 = 2.0f * q->x1 / W - 1, y0 = 2.0f * q->y0 / H - 1, y1 = 2.0f * q->y1 / H - 1;
		const float p[6][2] = {{x0, y0}, {x1, y0}, {x0, y1}, {x1, y0}, {x1, y1}, {x0, y1}};
		for (int i = 0; i < 6; i++) {
			float *o = v + (j * 6 + i) * 4;
			o[0] = p[i][0], o[1] = p[i][1], o[2] = 0, o[3] = 1;
		}
	}
	VkRenderPass rp = make_pass(k->targets);
	VkImageView views[4];
	for (unsigned i = 0; i < k->targets; i++)
		views[i] = t[i].view;
	VkFramebufferCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = rp,
				       .attachmentCount = k->targets, .pAttachments = views, .width = W, .height = H, .layers = 1};
	VkFramebuffer fb;
	CHECK(vkCreateFramebuffer(dev, &fci, NULL, &fb));
	VkPipelineLayoutCreateInfo lci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	VkPipelineLayout layout;
	CHECK(vkCreatePipelineLayout(dev, &lci, NULL, &layout));
	bool expect_refusal = k->targets > max_targets;
	VkPipeline pipe;
	VkResult pr = make_pipeline(k, rp, layout, &pipe);
	printf("case=%s targets=%u jobs=%u row_pitch=%llu%s\n", k->name, k->targets, JOBS,
	       (unsigned long long)t[0].row_pitch, k->inferred ? " inferred=register-form-outputs" : "");
	if (pr != VK_SUCCESS) {
		printf("  pipeline refused (%d)%s\n", pr, expect_refusal ? " as expected" : "");
		return expect_refusal ? 0 : 4;
	}
	VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
					   .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
	VkCommandBuffer cb;
	CHECK(vkAllocateCommandBuffers(dev, &cai, &cb));
	int worst = 0;
	for (unsigned j = 0; j < JOBS; j++) {
		CHECK(vkResetCommandBuffer(cb, 0));
		VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
					       .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
		CHECK(vkBeginCommandBuffer(cb, &bi));
		VkClearValue cv[4];
		for (unsigned i = 0; i < k->targets; i++)
			memcpy(cv[i].color.float32, clear_c[(i + j) % 4], 16);
		VkRenderPassBeginInfo rbi = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = rp,
					     .framebuffer = fb, .renderArea = {{0, 0}, {W, H}}, .clearValueCount = k->targets,
					     .pClearValues = cv};
		vkCmdBeginRenderPass(cb, &rbi, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
		VkDeviceSize off = 0;
		vkCmdBindVertexBuffers(cb, 0, 1, &vb, &off);
		VkViewport vpt = {0, 0, W, H, 0, 1};
		VkRect2D sc = {{0, 0}, {W, H}};
		vkCmdSetViewport(cb, 0, 1, &vpt);
		vkCmdSetScissor(cb, 0, 1, &sc);
		vkCmdDraw(cb, 6, 1, j * 6, 0);
		vkCmdEndRenderPass(cb);
		VkResult r = vkEndCommandBuffer(cb);
		const struct rect *q = &quads[j];
		printf("  job=%u quad=%u,%u-%u,%u", j, q->x0, q->y0, q->x1, q->y1);
		if (r != VK_SUCCESS) {
			printf(" refused at EndCommandBuffer (%d)%s\n", r, expect_refusal ? " as expected" : "");
			return expect_refusal ? 0 : 4;
		}
		if (expect_refusal) {
			printf(" recorded although the device advertises %u colour attachments\n", max_targets);
			return 4;
		}
		VkFence f;
		VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
		CHECK(vkCreateFence(dev, &fi, NULL, &f));
		VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
		double t0 = now_ms();
		r = vkQueueSubmit(queue, 1, &si, f);
		if (r == VK_SUCCESS)
			r = vkWaitForFences(dev, 1, &f, VK_TRUE, 30000000000ull);
		double ms = now_ms() - t0;
		vkDestroyFence(dev, f, NULL);
		if (r != VK_SUCCESS) {
			printf(" ms=%.1f result=FAILED (VkResult %d: -8 refused, -4 device lost, 2 fence timeout)\n", ms, r);
			return 4;
		}
		if (offline) {
			printf(" ms=%.1f result=ACCEPTED\n", ms);
			continue;
		}
		printf(" ms=%.1f", ms);
		int jr = 0;
		for (unsigned i = 0; i < k->targets; i++) {
			uint32_t clear = bgra(clear_c[(i + j) % 4]), out = out_w[i];
			uint64_t bad = 0, sentinel = 0, foreign = 0, gb = guard_bad(&t[i]);
			int fx = -1, fy = -1;
			uint32_t got0 = 0, want0 = 0;
			for (uint32_t y = 0; y < H; y++)
				for (uint32_t x = 0; x < W; x++) {
					uint32_t got, want = x >= q->x0 && x < q->x1 && y >= q->y0 && y < q->y1 ? out : clear;
					memcpy(&got, t[i].map + t[i].img_offset + y * t[i].row_pitch + 4 * x, 4);
					if (got == want)
						continue;
					sentinel += (got & 0xffff0000u) == 0x5a5a0000u;
					for (unsigned o = 0; o < 4; o++)
						foreign += o != i && (got == out_w[o] || got == bgra(clear_c[(o + j) % 4]));
					if (!bad++)
						fx = x, fy = y, got0 = got, want0 = want;
				}
			printf(" rt%u:bad=%llu", i, (unsigned long long)bad);
			if (bad)
				printf(",sentinel=%llu,foreign=%llu,first=%d,%d,got=%08x,want=%08x", (unsigned long long)sentinel,
				       (unsigned long long)foreign, fx, fy, got0, want0);
			printf(",guard=%llu", (unsigned long long)gb);
			if (bad)
				jr = jr > 1 ? jr : 1;
			if (gb)
				jr = 2;
			if (outdir) {
				char name[256];
				snprintf(name, sizeof(name), "rt-%s-j%u-%u-after.raw", k->name, j, i);
				dump(&t[i], name);
			}
		}
		printf(" result=%s\n", jr == 2 ? "GUARD" : jr ? "MISMATCH" : "EXACT");
		worst = jr > worst ? jr : worst;
	}
	vkFreeCommandBuffers(dev, pool, 1, &cb);
	vkDestroyPipeline(dev, pipe, NULL);
	vkDestroyPipelineLayout(dev, layout, NULL);
	vkDestroyFramebuffer(dev, fb, NULL);
	vkDestroyRenderPass(dev, rp, NULL);
	vkDestroyBuffer(dev, vb, NULL);
	vkFreeMemory(dev, bmem, NULL);
	for (unsigned i = 0; i < k->targets; i++) {
		vkDestroyImageView(dev, t[i].view, NULL);
		vkDestroyImage(dev, t[i].img, NULL);
		vkFreeMemory(dev, t[i].mem, NULL);
	}
	return worst;
}

int main(int argc, char **argv)
{
	const char *only = NULL;
	int expect_max = 0;
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--offline"))
			offline = true;
		else if (!strcmp(argv[i], "--out") && i + 1 < argc)
			outdir = argv[++i];
		else if (!strcmp(argv[i], "--case") && i + 1 < argc)
			only = argv[++i];
		else if (!strcmp(argv[i], "--expect-max") && i + 1 < argc)
			expect_max = atoi(argv[++i]);
		else {
			fprintf(stderr, "usage: %s [--offline] [--out DIR] [--case NAME[,NAME...]] [--expect-max N]\n", argv[0]);
			return 3;
		}
	}
	setvbuf(stdout, NULL, _IOLBF, 0);
	VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "mrt-vk", .apiVersion = VK_API_VERSION_1_0};
	VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
	VkInstance inst;
	CHECK(vkCreateInstance(&ici, NULL, &inst));
	uint32_t n = 1;
	VkPhysicalDevice pd;
	VkResult er = vkEnumeratePhysicalDevices(inst, &n, &pd);
	if ((er != VK_SUCCESS && er != VK_INCOMPLETE) || !n) {
		fprintf(stderr, "mrt-vk: no physical device (%d)\n", er);
		return 3;
	}
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(pd, &props);
	vkGetPhysicalDeviceMemoryProperties(pd, &memprops);
	unsigned max = props.limits.maxColorAttachments;
	printf("device=%s offline=%d maxColorAttachments=%u maxFragmentOutputAttachments=%u\n", props.deviceName, offline,
	       max, props.limits.maxFragmentOutputAttachments);
	if (expect_max && (int)max != expect_max) {
		printf("result=SETUP (maxColorAttachments %u, expected %d)\n", max, expect_max);
		return 3;
	}
	if (!expect_max && max < 4) {
		printf("result=SETUP (maxColorAttachments %u: not the MRT-capable ICD, or the option is off; see --expect-max)\n", max);
		return 3;
	}
	float prio = 1.0f;
	VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1, .pQueuePriorities = &prio};
	VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci};
	CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
	vkGetDeviceQueue(dev, 0, 0, &queue);
	VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
				       .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
	CHECK(vkCreateCommandPool(dev, &pci, NULL, &pool));
	int worst = 0, ran = 0;
	char list[256];
	snprintf(list, sizeof(list), "%s", only ? only : "one,mrt2,mrt4"); /* the inferred case runs only when named */
	for (char *save = NULL, *name = strtok_r(list, ",", &save); name; name = strtok_r(NULL, ",", &save)) {
		const struct tcase *k = NULL;
		for (unsigned c = 0; c < sizeof(cases) / sizeof(cases[0]); c++)
			if (!strcmp(name, cases[c].name))
				k = &cases[c];
		if (!k)
			return fprintf(stderr, "mrt-vk: no case %s\n", name), 3;
		int r = run_case(k, expect_max ? (unsigned)expect_max : max);
		ran++;
		if (r == 4) {
			printf("result=REFUSED\n");
			return 4;
		}
		worst = r > worst ? r : worst;
	}
	if (!ran)
		return fprintf(stderr, "mrt-vk: no case\n"), 3;
	vkDestroyCommandPool(dev, pool, NULL);
	vkDestroyDevice(dev, NULL);
	vkDestroyInstance(inst, NULL);
	printf("result=%s\n", offline ? "PASS-OFFLINE" : worst == 0 ? "PASS" : worst == 2 ? "GUARD" : "MISMATCH");
	return offline ? 0 : worst;
}
