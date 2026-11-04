/* SPDX-License-Identifier: MIT */
/* msaa-vk: 4x multisampling, Vulkan with VK_KHR_dynamic_rendering, checked against a CPU model.
 * Ported from the Apple A9 (G5P) Linux port's fixture, where all 8 cases below are exact on hardware.
 *
 * G9P es3-msaa implements one, resolve, depth, store and resolvecmd (A10 untested).
 * On branches without MSAA, run with --expect-no-msaa. Coverage/sample-mask cases are not implemented.
 *
 * Every case is one render job (dynamic rendering, one submit + fence wait) at 64x64 B8G8R8A8: CLEAR blue, then
 * the triangle (-.75,-.75) (.75,-.75) (0,.75) in red, with a negative-height viewport (Metal's y, so the pixels
 * are those of Apple's iOS msaa4 capture). The result is compared byte for byte with a CPU model: the standard 4x
 * positions (6,2) (14,6) (2,10) (10,14)/16, a sample covered when inside the triangle, resolve = (sum + 2) / 4 per
 * byte. On the A9 that model was checked equal to Apple's own GPU results (raw samples, resolved pixels, the
 * sample-mask and alpha-to-coverage scenes with mask 0x9).
 *   one         one sample, the colour path (control)
 *   resolve     4x, resolved into a one-sample linear view at the end of the pass (storeOp DONT_CARE)
 *   store       4x, STORE + a resolve view: the samples are stored and resolved
 *   resolvecmd  4x stored, then vkCmdResolveImage
 *   depth       4x colour + 4x D32F on chip: red triangle z=.5, then a green full quad z=.75 Less: per-sample depth
 *   a2c         alpha .5 with alpha-to-coverage: samples 0 and 3 (what Apple's A9 keeps), alpha stored as 127
 *   smask       pSampleMask 0x5: samples 0 and 2
 *   smaskshader gl_SampleMask[0] = 5 in the fragment program: the same pixels
 * store and resolvecmd also read the 4x image's memory directly (raw_bad=): that check assumes the A9's layout
 * (16 bytes per pixel, sample s at +4s, Morton order with y in bit 0) and does not decide the result.
 *
 *   msaa-vk [--offline] [--out DIR] [--case NAME[,NAME...]] [--expect-no-msaa]
 *
 * One line per case and result=. Exit 0 all exact, 1 a pixel mismatch, 3 setup error (or a 4x case asked of a
 * device that offers no 4 samples: result=NOT-OFFERED), 4 a job refused or failed (-8 refused, -4 device lost,
 * 2 fence timeout). --case takes whole names. --offline (a driver that executes nothing): every job must be
 * accepted, no pixels. --expect-no-msaa: the device must not offer 4 samples; the 4x cases are skipped and the
 * control still runs.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#include "color.frag.h"
#include "color.vert.h"
#include "smask.frag.h"

#define CHECK(x)                                                                           \
	do {                                                                               \
		VkResult r_ = (x);                                                         \
		if (r_ != VK_SUCCESS) {                                                    \
			fprintf(stderr, "msaa-vk: %s = %d (line %d)\n", #x, r_, __LINE__); \
			exit(3);                                                           \
		}                                                                          \
	} while (0)

#define N 64
static VkDevice dev;
static VkPhysicalDevice phys;
static VkQueue queue;
static VkCommandPool pool;
static VkPhysicalDeviceMemoryProperties memprops;
static bool offline;
static const char *outdir;
static PFN_vkCmdBeginRendering begin_rendering;
static PFN_vkCmdEndRendering end_rendering;

static const int pos[4][2] = {{6, 2}, {14, 6}, {2, 10}, {10, 14}};
static const uint8_t BLUE[4] = {255, 0, 0, 255}, RED[4] = {0, 0, 255, 255}, GREEN[4] = {0, 255, 0, 255},
		     HALF_RED[4] = {0, 0, 255, 127};

enum kind { ONE, RESOLVE, STORE, RESOLVECMD, DEPTH, A2C, SMASK, SMASKSHADER };
static const char *names[] = {"one", "resolve", "store", "resolvecmd", "depth", "a2c", "smask", "smaskshader"};

/* 16 * edge function, integer: no sample sits on an edge of this triangle */
static bool inside(int px16, int py16)
{
	static const int t[3][2] = {{8 * 16, 56 * 16}, {56 * 16, 56 * 16}, {32 * 16, 8 * 16}};
	int s[3];
	for (int i = 0; i < 3; i++) {
		const int *a = t[i], *b = t[(i + 1) % 3];
		s[i] = (b[0] - a[0]) * (py16 - a[1]) - (b[1] - a[1]) * (px16 - a[0]);
	}
	return (s[0] < 0 && s[1] < 0 && s[2] < 0) || (s[0] > 0 && s[1] > 0 && s[2] > 0);
}

static unsigned cover(int x, int y, bool one)
{
	if (one)
		return inside(x * 16 + 8, y * 16 + 8) ? 1 : 0;
	unsigned m = 0;
	for (int s = 0; s < 4; s++)
		m |= inside(x * 16 + pos[s][0], y * 16 + pos[s][1]) << s;
	return m;
}

/* per-pixel sample texels of a case, and its resolve */
static void model(enum kind k, int x, int y, uint8_t smp[4][4])
{
	unsigned c = cover(x, y, k == ONE), keep = k == A2C ? 0x9 : (k == SMASK || k == SMASKSHADER) ? 0x5 : 0xf;
	const uint8_t *fg = k == A2C ? HALF_RED : RED, *bg = k == DEPTH ? GREEN : BLUE;
	for (int s = 0; s < 4; s++)
		memcpy(smp[s], (k == ONE ? c & 1 : (c & keep) >> s & 1) ? fg : bg, 4);
}

static void resolved(enum kind k, int x, int y, uint8_t out[4])
{
	uint8_t s[4][4];
	model(k, x, y, s);
	for (int b = 0; b < 4; b++)
		out[b] = k == ONE ? s[0][b] : (uint8_t)((s[0][b] + s[1][b] + s[2][b] + s[3][b] + 2) / 4);
}

static uint32_t twiddle(uint32_t x, uint32_t y)
{
	uint32_t r = 0;
	for (int b = 0; b < 6; b++)
		r |= ((y >> b) & 1) << (2 * b) | ((x >> b) & 1) << (2 * b + 1);
	return r;
}

static uint32_t mem_type(uint32_t bits)
{
	for (uint32_t i = 0; i < memprops.memoryTypeCount; i++)
		if ((bits & (1u << i)) && (memprops.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
			return i;
	fprintf(stderr, "msaa-vk: no host-visible memory type\n");
	exit(3);
}

struct surface {
	VkImage image;
	VkImageView view;
	VkDeviceMemory mem;
	uint8_t *map;
	VkDeviceSize size;
	uint32_t pitch;
};

static void make_surface(struct surface *s, VkFormat f, VkSampleCountFlagBits samples, bool linear)
{
	bool depth = f == VK_FORMAT_D32_SFLOAT;
	VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
		.format = f, .extent = {N, N, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = samples,
		.tiling = linear ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL,
		.usage = depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT :
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
			VK_IMAGE_USAGE_TRANSFER_DST_BIT};
	CHECK(vkCreateImage(dev, &ci, NULL, &s->image));
	VkMemoryRequirements req;
	vkGetImageMemoryRequirements(dev, s->image, &req);
	s->size = req.size;
	VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
		.memoryTypeIndex = mem_type(req.memoryTypeBits)};
	CHECK(vkAllocateMemory(dev, &ai, NULL, &s->mem));
	CHECK(vkBindImageMemory(dev, s->image, s->mem, 0));
	CHECK(vkMapMemory(dev, s->mem, 0, VK_WHOLE_SIZE, 0, (void **)&s->map));
	memset(s->map, 0xa5, req.size);
	s->pitch = N * 4;
	if (linear) {
		VkSubresourceLayout l;
		vkGetImageSubresourceLayout(dev, s->image, &(VkImageSubresource){VK_IMAGE_ASPECT_COLOR_BIT, 0, 0}, &l);
		s->pitch = (uint32_t)l.rowPitch;
	}
	VkImageViewCreateInfo vci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = s->image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D, .format = f,
		.subresourceRange = {depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
	CHECK(vkCreateImageView(dev, &vci, NULL, &s->view));
}

static VkResult make_pipeline(enum kind k, VkPipeline *out)
{
	VkShaderModuleCreateInfo vsi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = sizeof(spv_color_vert), .pCode = spv_color_vert};
	VkShaderModuleCreateInfo fsi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = k == SMASKSHADER ? sizeof(spv_smask_frag) : sizeof(spv_color_frag),
		.pCode = k == SMASKSHADER ? spv_smask_frag : spv_color_frag};
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
		.polygonMode = VK_POLYGON_MODE_FILL, .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1};
	VkSampleMask mask = 0x5;
	VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = k == ONE ? VK_SAMPLE_COUNT_1_BIT : VK_SAMPLE_COUNT_4_BIT,
		.alphaToCoverageEnable = k == A2C, .pSampleMask = k == SMASK ? &mask : NULL};
	VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = k == DEPTH, .depthWriteEnable = k == DEPTH, .depthCompareOp = VK_COMPARE_OP_LESS};
	VkPipelineColorBlendAttachmentState cba = {.colorWriteMask = 0xf};
	VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1, .pAttachments = &cba};
	VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2, .pDynamicStates = dyn};
	const VkFormat color = VK_FORMAT_B8G8R8A8_UNORM;
	VkPipelineRenderingCreateInfo ren = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
		.colorAttachmentCount = 1, .pColorAttachmentFormats = &color,
		.depthAttachmentFormat = k == DEPTH ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_UNDEFINED};
	VkPipelineLayoutCreateInfo lci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	VkPipelineLayout layout;
	CHECK(vkCreatePipelineLayout(dev, &lci, NULL, &layout));
	VkGraphicsPipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ren,
		.stageCount = 2, .pStages = st, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
		.pViewportState = &vp, .pRasterizationState = &rs, .pMultisampleState = &ms,
		.pDepthStencilState = k == DEPTH ? &ds : NULL, .pColorBlendState = &cb, .pDynamicState = &dy,
		.layout = layout};
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

static void dump(const uint8_t *p, size_t n, const char *k, const char *what)
{
	if (!outdir)
		return;
	char path[512];
	snprintf(path, sizeof(path), "%s/%s-%s.bin", outdir, k, what);
	FILE *f = fopen(path, "wb");
	if (f) {
		fwrite(p, 1, n, f);
		fclose(f);
	}
}

static int run_case(enum kind k)
{
	const bool ms = k != ONE;
	struct surface msimg = {0}, res, z = {0};
	make_surface(&res, VK_FORMAT_B8G8R8A8_UNORM, VK_SAMPLE_COUNT_1_BIT, true);
	if (ms)
		make_surface(&msimg, VK_FORMAT_B8G8R8A8_UNORM, VK_SAMPLE_COUNT_4_BIT, false);
	if (k == DEPTH)
		make_surface(&z, VK_FORMAT_D32_SFLOAT, VK_SAMPLE_COUNT_4_BIT, false);
	VkPipeline pipe, green = VK_NULL_HANDLE;
	VkResult r = make_pipeline(k, &pipe);
	if (r == VK_SUCCESS && k == DEPTH)
		r = make_pipeline(DEPTH, &green);
	if (r != VK_SUCCESS) {
		printf("case=%s pipeline=%d\nresult=REFUSED\n", names[k], r);
		return 4;
	}
	const float a = k == A2C ? .5f : 1;
	const float v[3 * 8 + 6 * 8] = {
		-.75f, -.75f, .5f, 1, 1, 0, 0, a, .75f, -.75f, .5f, 1, 1, 0, 0, a, 0, .75f, .5f, 1, 1, 0, 0, a,
		-1, -1, .75f, 1, 0, 1, 0, 1, 1, -1, .75f, 1, 0, 1, 0, 1, 1, 1, .75f, 1, 0, 1, 0, 1,
		-1, -1, .75f, 1, 0, 1, 0, 1, 1, 1, .75f, 1, 0, 1, 0, 1, -1, 1, .75f, 1, 0, 1, 0, 1};
	VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = sizeof(v),
		.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT};
	VkBuffer vb;
	CHECK(vkCreateBuffer(dev, &bci, NULL, &vb));
	VkMemoryRequirements req;
	vkGetBufferMemoryRequirements(dev, vb, &req);
	VkDeviceMemory vm;
	VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
		.memoryTypeIndex = mem_type(req.memoryTypeBits)};
	CHECK(vkAllocateMemory(dev, &ai, NULL, &vm));
	CHECK(vkBindBufferMemory(dev, vb, vm, 0));
	void *p;
	CHECK(vkMapMemory(dev, vm, 0, VK_WHOLE_SIZE, 0, &p));
	memcpy(p, v, sizeof(v));

	VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
	VkCommandBuffer cmd;
	CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
	VkCommandBufferBeginInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	CHECK(vkBeginCommandBuffer(cmd, &cbi));
	const bool pbe_resolve = k != ONE && k != STORE && k != RESOLVECMD;
	VkRenderingAttachmentInfo color = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
		.imageView = ms ? msimg.view : res.view, .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		.resolveMode = pbe_resolve || k == STORE ? VK_RESOLVE_MODE_AVERAGE_BIT : VK_RESOLVE_MODE_NONE,
		.resolveImageView = pbe_resolve || k == STORE ? res.view : VK_NULL_HANDLE,
		.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		.storeOp = pbe_resolve ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE,
		.clearValue.color.float32 = {0, 0, 1, 1}};
	VkRenderingAttachmentInfo zs = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = z.view,
		.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE, .clearValue.depthStencil = {1.0f, 0}};
	VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {N, N}},
		.layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &color,
		.pDepthAttachment = k == DEPTH ? &zs : NULL};
	begin_rendering(cmd, &ri);
	VkDeviceSize zero = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &zero);
	vkCmdSetViewport(cmd, 0, 1, &(VkViewport){0, N, N, -N, 0, 1});
	vkCmdSetScissor(cmd, 0, 1, &(VkRect2D){{0, 0}, {N, N}});
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
	vkCmdDraw(cmd, 3, 1, 0, 0);
	if (k == DEPTH) {
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, green);
		vkCmdDraw(cmd, 6, 1, 3, 0);
	}
	end_rendering(cmd);
	if (k == RESOLVECMD) {
		VkImageResolve reg = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
			.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .extent = {N, N, 1}};
		vkCmdResolveImage(cmd, msimg.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, res.image,
				  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);
	}
	r = vkEndCommandBuffer(cmd);
	double t0 = now_ms();
	if (r == VK_SUCCESS) {
		VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
		VkFence fence;
		CHECK(vkCreateFence(dev, &fci, NULL, &fence));
		VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd};
		r = vkQueueSubmit(queue, 1, &si, fence);
		if (r == VK_SUCCESS)
			r = vkWaitForFences(dev, 1, &fence, VK_TRUE, 60ull * 1000 * 1000 * 1000);
		vkDestroyFence(dev, fence, NULL);
	}
	printf("case=%s samples=%u job=%d %s ms=%.1f\n", names[k], ms ? 4 : 1, r, r == VK_SUCCESS ? "accepted" : "failed",
	       now_ms() - t0);
	if (r != VK_SUCCESS) {
		printf("result=FAILED vk=%d\n", r);
		return 4;
	}
	if (offline) {
		printf("result=ACCEPTED (offline: no pixels)\n");
		return 0;
	}
	unsigned bad = 0, sbad = 0, first = ~0u;
	for (int y = 0; y < N; y++)
		for (int x = 0; x < N; x++) {
			uint8_t want[4];
			resolved(k, x, y, want);
			if (memcmp(res.map + (size_t)y * res.pitch + 4 * x, want, 4)) {
				if (first == ~0u)
					first = y * N + x;
				bad++;
			}
		}
	if (k == STORE || k == RESOLVECMD)
		for (int y = 0; y < N; y++)
			for (int x = 0; x < N; x++) {
				uint8_t smp[4][4];
				model(k, x, y, smp);
				sbad += memcmp(msimg.map + twiddle(x, y) * 16, smp, 16) != 0;
			}
	uint8_t rows[N * N * 4];
	for (int y = 0; y < N; y++)
		memcpy(rows + y * N * 4, res.map + (size_t)y * res.pitch, N * 4);
	dump(rows, sizeof(rows), names[k], "resolved");
	if (ms)
		dump(msimg.map, N * N * 16, names[k], "samples");
	if (first != ~0u) {
		const uint8_t *g = res.map + (size_t)(first / N) * res.pitch + 4 * (first % N);
		uint8_t w[4];
		resolved(k, first % N, first / N, w);
		printf("first mismatch (%u,%u): got %02x%02x%02x%02x want %02x%02x%02x%02x\n", first % N, first / N, g[0],
		       g[1], g[2], g[3], w[0], w[1], w[2], w[3]);
	}
	printf("result=%s bad=%u raw_bad=%u%s\n", bad ? "MISMATCH" : "EXACT", bad, sbad,
	       k == STORE || k == RESOLVECMD ? " (raw_bad: the A9 4x layout, informational)" : "");
	return bad ? 1 : 0;
}

/* --case: whole comma-separated names ("resolve" does not select "resolvecmd"). */
static bool wanted(const char *only, const char *name)
{
	size_t n = strlen(name);
	for (const char *p = only; p && *p; p = strchr(p, ',') ? strchr(p, ',') + 1 : NULL)
		if (!strncmp(p, name, n) && (p[n] == ',' || !p[n]))
			return true;
	return false;
}

int main(int argc, char **argv)
{
	const char *only = NULL;
	bool expect_no = false;
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--offline"))
			offline = true;
		else if (!strcmp(argv[i], "--expect-no-msaa"))
			expect_no = true;
		else if (!strcmp(argv[i], "--out") && i + 1 < argc)
			outdir = argv[++i];
		else if (!strcmp(argv[i], "--case") && i + 1 < argc)
			only = argv[++i];
		else {
			fprintf(stderr, "usage: msaa-vk [--offline] [--out DIR] [--case NAME[,NAME...]] [--expect-no-msaa]\n");
			return 3;
		}
	}
	VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_0};
	VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
	VkInstance inst;
	CHECK(vkCreateInstance(&ici, NULL, &inst));
	uint32_t n = 1;
	VkResult er = vkEnumeratePhysicalDevices(inst, &n, &phys);
	if ((er != VK_SUCCESS && er != VK_INCOMPLETE) || !n) {
		fprintf(stderr, "msaa-vk: no device\n");
		return 3;
	}
	vkGetPhysicalDeviceMemoryProperties(phys, &memprops);
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(phys, &props);
	bool has4 = props.limits.framebufferColorSampleCounts & VK_SAMPLE_COUNT_4_BIT;
	printf("device=%s framebufferColorSampleCounts=0x%x depth=0x%x standardSampleLocations=%u\n", props.deviceName,
	       props.limits.framebufferColorSampleCounts, props.limits.framebufferDepthSampleCounts,
	       props.limits.standardSampleLocations);
	if (expect_no && has4) {
		printf("result=UNEXPECTED (4 samples offered, but --expect-no-msaa was given)\n");
		return 1;
	}
	float prio = 1;
	VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1,
		.pQueuePriorities = &prio};
	const char *ext[] = {"VK_KHR_dynamic_rendering"};
	VkPhysicalDeviceDynamicRenderingFeatures dr = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
		.dynamicRendering = VK_TRUE};
	VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &dr, .queueCreateInfoCount = 1,
		.pQueueCreateInfos = &qci, .enabledExtensionCount = 1, .ppEnabledExtensionNames = ext};
	CHECK(vkCreateDevice(phys, &dci, NULL, &dev));
	begin_rendering = (PFN_vkCmdBeginRendering)vkGetDeviceProcAddr(dev, "vkCmdBeginRenderingKHR");
	end_rendering = (PFN_vkCmdEndRendering)vkGetDeviceProcAddr(dev, "vkCmdEndRenderingKHR");
	if (!begin_rendering || !end_rendering) {
		fprintf(stderr, "msaa-vk: no dynamic rendering\n");
		return 3;
	}
	vkGetDeviceQueue(dev, 0, 0, &queue);
	VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
	CHECK(vkCreateCommandPool(dev, &pci, NULL, &pool));
	int worst = 0;
	for (int k = ONE; k <= SMASKSHADER; k++) {
		if (only && !wanted(only, names[k]))
			continue;
		if (k != ONE && expect_no) {
			printf("case=%s skipped (no 4x)\n", names[k]);
			continue;
		}
		if (k != ONE && !has4) {
			/* Not a pass: the device reports no 4x colour sample count (use --expect-no-msaa for that build). */
			printf("case=%s samples=4 not offered (framebufferColorSampleCounts=0x%x)\nresult=NOT-OFFERED\n", names[k],
			       props.limits.framebufferColorSampleCounts);
			if (worst < 3)
				worst = 3;
			continue;
		}
		int rc = run_case((enum kind)k);
		if (rc > worst)
			worst = rc;
	}
	printf("result=%s\n", worst ? "FAIL" : "PASS");
	return worst;
}
