// SPDX-License-Identifier: MIT
/*
 * xfb-vk: transform feedback (VK_EXT_transform_feedback) of a known vertex stream, checked byte for byte.
 *
 * Plain Vulkan 1.1 plus VK_EXT_transform_feedback (no driver dependency). Each case is one render job on a 64x64
 * linear B8G8R8A8_UNORM target with guards: a CLEAR pass, then quads (two triangles each, a triangle list) drawn with a
 * flat colour while the vertex shader captures two buffers:
 *   buffer 0: cap_a = 2 * position + gl_VertexIndex, 16 bytes of every 32-byte record (the other 16 stay untouched);
 *   buffer 1: cap_b = (gl_VertexIndex, position.x, 1, -position.y), packed 16-byte records.
 * Both are bound at a non-zero offset inside a larger buffer filled with 0xa5. The CPU model is exact: the captured
 * records of the whole primitives that fit (the buffer with the least room decides), in stream order, and nothing
 * else changed; the counter buffer holds the bytes written from each binding's offset (vkCmdEndTransformFeedbackEXT);
 * the pixels are the quads' colour on the clear colour.
 *
 *   rec       one draw (2 triangles), room for all of it
 *   two       two draws in one Begin/End: the second's records follow the first's
 *   overflow  two draws, buffer 0 has room for 9 records: the first draw fits, the second captures only its first
 *             triangle (buffer 1 too: whole primitives, the least room decides), nothing past the binding
 *   resume    two Begin/End pairs (pause and resume through the counter buffer, as Zink does): records contiguous
 *   query     overflow's draws inside a stream query (VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT, reset with
 *             vkCmdResetQueryPool outside the pass): primitives written 3, needed 4, read both with
 *             vkGetQueryPoolResults (64-bit, with availability) and vkCmdCopyQueryPoolResults (32-bit, with
 *             availability, into a buffer after the pass)
 *   discard   rec with rasterizerDiscardEnable: the same capture, no pixel drawn (GL_RASTERIZER_DISCARD)
 *
 *   xfb-vk [--offline] [--case NAME[,NAME...]]
 * Output: one line per case, "case=NAME ... pixels=0 buf0=0 buf1=0 counters=0 query=0 guard=0 result=EXACT" (or MISMATCH with
 * the first difference), then "result=PASS" (exit 0), MISMATCH (1), GUARD (2); 3 setup error, 4 refused or failed.
 * --offline (a driver that executes nothing): "result=ACCEPTED" per case and "result=PASS-OFFLINE".
 */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "xfb-spirv.h"

#define CHECK(x)                                                                           \
	do {                                                                               \
		VkResult r_ = (x);                                                         \
		if (r_ != VK_SUCCESS) {                                                    \
			fprintf(stderr, "xfb-vk: %s = %d (line %d)\n", #x, r_, __LINE__); \
			exit(3);                                                           \
		}                                                                          \
	} while (0)

#define W 64
#define H 64
#define GUARD 16384u
#define STRIDE0 32u
#define STRIDE1 16u
#define OFF0 64u  /* binding offsets inside the buffers */
#define OFF1 48u
#define CTR_OFF0 8u /* counter offsets inside the counter buffer */
#define CTR_OFF1 20u
#define BUF_BYTES 1024u

static VkDevice dev;
static VkQueue queue;
static VkCommandPool pool;
static VkPhysicalDeviceMemoryProperties memprops;
static bool offline;
static PFN_vkCmdBindTransformFeedbackBuffersEXT bind_xfb;
static PFN_vkCmdBeginTransformFeedbackEXT begin_xfb;
static PFN_vkCmdEndTransformFeedbackEXT end_xfb;
static PFN_vkCmdBeginQueryIndexedEXT begin_query;
static PFN_vkCmdEndQueryIndexedEXT end_query;

struct quad { uint32_t x0, y0, x1, y1; };
struct tcase {
	const char *name;
	unsigned draws;
	struct quad q[2];
	uint32_t room0, room1; /* records the bindings hold */
	bool resume;           /* one Begin/End per draw, resumed through the counters */
	bool query, discard;
};

static const struct tcase cases[] = {
	{"rec", 1, {{16, 8, 40, 32}}, 6, 6, false, false, false},
	{"two", 2, {{8, 8, 24, 40}, {32, 24, 56, 56}}, 12, 12, false, false, false},
	{"overflow", 2, {{8, 8, 24, 40}, {32, 24, 56, 56}}, 9, 12, false, false, false},
	{"resume", 2, {{8, 8, 24, 40}, {32, 24, 56, 56}}, 12, 12, true, false, false},
	{"query", 2, {{8, 8, 24, 40}, {32, 24, 56, 56}}, 9, 12, false, true, false},
	{"discard", 1, {{16, 8, 40, 32}}, 6, 6, false, false, true},
};
static const float clear_c[4] = {0.0f, 0.0f, 0.0f, 1.0f};
static const float quad_c[4] = {0.2f, 0.4f, 0.6f, 0.8f};

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
		if ((bits & (1u << i)) && (memprops.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
		    (memprops.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
			return i;
	fprintf(stderr, "xfb-vk: no host-visible coherent memory type\n");
	exit(3);
}

struct buf { VkBuffer b; VkDeviceMemory m; uint8_t *map; VkDeviceSize size; };
static void make_buf(struct buf *o, VkDeviceSize size, VkBufferUsageFlags usage, uint8_t fill)
{
	VkBufferCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage};
	CHECK(vkCreateBuffer(dev, &ci, NULL, &o->b));
	VkMemoryRequirements mr;
	vkGetBufferMemoryRequirements(dev, o->b, &mr);
	VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
				   .memoryTypeIndex = mem_type(mr.memoryTypeBits)};
	CHECK(vkAllocateMemory(dev, &ai, NULL, &o->m));
	CHECK(vkMapMemory(dev, o->m, 0, VK_WHOLE_SIZE, 0, (void **)&o->map));
	memset(o->map, fill, size);
	o->size = size;
	CHECK(vkBindBufferMemory(dev, o->b, o->m, 0));
}
static void free_buf(struct buf *o)
{
	vkDestroyBuffer(dev, o->b, NULL);
	vkFreeMemory(dev, o->m, NULL);
}

struct target {
	VkImage img;
	VkImageView view;
	VkDeviceMemory mem;
	uint8_t *map;
	VkDeviceSize size, row_pitch, img_offset;
};

static void make_target(struct target *t)
{
	VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
		.format = VK_FORMAT_B8G8R8A8_UNORM, .extent = {W, H, 1}, .mipLevels = 1, .arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_LINEAR,
		.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
	CHECK(vkCreateImage(dev, &ii, NULL, &t->img));
	VkMemoryRequirements mr;
	vkGetImageMemoryRequirements(dev, t->img, &mr);
	if (GUARD % mr.alignment) {
		fprintf(stderr, "xfb-vk: image alignment %llu does not divide the guard\n", (unsigned long long)mr.alignment);
		exit(3);
	}
	t->size = GUARD + mr.size + GUARD;
	VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = t->size,
				   .memoryTypeIndex = mem_type(mr.memoryTypeBits)};
	CHECK(vkAllocateMemory(dev, &ai, NULL, &t->mem));
	CHECK(vkMapMemory(dev, t->mem, 0, VK_WHOLE_SIZE, 0, (void **)&t->map));
	memset(t->map, 0xa5, t->size);
	CHECK(vkBindImageMemory(dev, t->img, t->mem, GUARD));
	VkSubresourceLayout lay;
	vkGetImageSubresourceLayout(dev, t->img, &(VkImageSubresource){VK_IMAGE_ASPECT_COLOR_BIT, 0, 0}, &lay);
	t->row_pitch = lay.rowPitch;
	t->img_offset = GUARD + lay.offset;
	for (uint32_t y = 0; y < H; y++)
		memset(t->map + t->img_offset + y * t->row_pitch, 0x5a, 4 * W);
	VkImageViewCreateInfo vci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = t->img,
		.viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_B8G8R8A8_UNORM,
		.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
	CHECK(vkCreateImageView(dev, &vci, NULL, &t->view));
}

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

static VkShaderModule module(const uint32_t *code, size_t size)
{
	VkShaderModuleCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = code};
	VkShaderModule m;
	CHECK(vkCreateShaderModule(dev, &ci, NULL, &m));
	return m;
}

/* the vertex stream: draw d's quad is vertices 6d..6d+5 (two triangles); positions are multiples of 1/32 */
static void quad_positions(const struct quad *q, float p[6][4])
{
	const float x0 = 2.0f * q->x0 / W - 1, x1 = 2.0f * q->x1 / W - 1, y0 = 2.0f * q->y0 / H - 1, y1 = 2.0f * q->y1 / H - 1;
	const float v[6][2] = {{x0, y0}, {x1, y0}, {x0, y1}, {x1, y0}, {x1, y1}, {x0, y1}};
	for (int i = 0; i < 6; i++)
		p[i][0] = v[i][0], p[i][1] = v[i][1], p[i][2] = 0, p[i][3] = 1;
}

/* compare the model with the buffer; count differing bytes, report the first */
static uint64_t diff(const char *what, const uint8_t *got, const uint8_t *want, size_t n, bool *first)
{
	uint64_t bad = 0;
	for (size_t i = 0; i < n; i++)
		if (got[i] != want[i] && !bad++ && *first) {
			printf(" first=%s+%zu got=%02x want=%02x", what, i, got[i], want[i]);
			*first = false;
		}
	return bad;
}

/* 0 exact, 1 mismatch, 2 guard, 4 refused/failed */
static int run_case(const struct tcase *k)
{
	struct target t;
	make_target(&t);

	struct buf vb, x0, x1, ctr;
	make_buf(&vb, 2 * 6 * 16, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 0);
	float pos[2][6][4];
	for (unsigned d = 0; d < k->draws; d++) {
		quad_positions(&k->q[d], pos[d]);
		memcpy(vb.map + d * sizeof(pos[d]), pos[d], sizeof(pos[d]));
	}
	make_buf(&x0, BUF_BYTES, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT, 0xa5);
	make_buf(&x1, BUF_BYTES, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT, 0xa5);
	make_buf(&ctr, 64, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT, 0xa5);

	VkPipelineLayoutCreateInfo plci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	VkPipelineLayout layout;
	CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &layout));
	VkAttachmentDescription a = {.format = VK_FORMAT_B8G8R8A8_UNORM, .samples = VK_SAMPLE_COUNT_1_BIT,
		.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, .finalLayout = VK_IMAGE_LAYOUT_GENERAL};
	VkAttachmentReference ref = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
	VkSubpassDescription sub = {.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS, .colorAttachmentCount = 1,
				    .pColorAttachments = &ref};
	VkRenderPassCreateInfo rpci = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, .attachmentCount = 1,
				       .pAttachments = &a, .subpassCount = 1, .pSubpasses = &sub};
	VkRenderPass rp;
	CHECK(vkCreateRenderPass(dev, &rpci, NULL, &rp));
	VkFramebufferCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = rp,
				       .attachmentCount = 1, .pAttachments = &t.view, .width = W, .height = H, .layers = 1};
	VkFramebuffer fb;
	CHECK(vkCreateFramebuffer(dev, &fci, NULL, &fb));

	VkShaderModule vs = module(spv_vs_xfb, sizeof(spv_vs_xfb));
	VkShaderModule fs = module(spv_fs_flat, sizeof(spv_fs_flat));
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
		.rasterizerDiscardEnable = k->discard, .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1};
	VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
						   .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
	VkPipelineColorBlendAttachmentState att = {.colorWriteMask = 0xf};
	VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
						  .attachmentCount = 1, .pAttachments = &att};
	VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	VkPipelineDynamicStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
					       .dynamicStateCount = 2, .pDynamicStates = dyn};
	VkGraphicsPipelineCreateInfo gci = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = 2,
		.pStages = st, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vp,
		.pRasterizationState = &rs, .pMultisampleState = &ms, .pColorBlendState = &cb, .pDynamicState = &ds,
		.layout = layout, .renderPass = rp};
	VkPipeline pipe;
	VkResult pr_res = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gci, NULL, &pipe);
	vkDestroyShaderModule(dev, vs, NULL);
	vkDestroyShaderModule(dev, fs, NULL);
	printf("case=%s draws=%u room=%u,%u%s%s%s", k->name, k->draws, k->room0, k->room1, k->resume ? " resume" : "",
	       k->query ? " query" : "", k->discard ? " discard" : "");
	if (pr_res != VK_SUCCESS) {
		printf(" result=FAILED (pipeline %d)\n", pr_res);
		return 4;
	}

	VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
					   .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
	VkCommandBuffer cmd;
	CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
	VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
				       .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
	CHECK(vkBeginCommandBuffer(cmd, &bi));
	VkQueryPool qp = VK_NULL_HANDLE;
	struct buf qb = {0};
	if (k->query) {
		VkQueryPoolCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
					     .queryType = VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT, .queryCount = 2};
		CHECK(vkCreateQueryPool(dev, &qci, NULL, &qp));
		make_buf(&qb, 64, VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0xa5);
		vkCmdResetQueryPool(cmd, qp, 0, 2);
	}
	VkClearValue cv;
	memcpy(cv.color.float32, clear_c, sizeof(clear_c));
	VkRenderPassBeginInfo rbi = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = rp, .framebuffer = fb,
				     .renderArea = {{0, 0}, {W, H}}, .clearValueCount = 1, .pClearValues = &cv};
	vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
	VkDeviceSize zero = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &vb.b, &zero);
	VkViewport vpt = {0, 0, W, H, 0, 1};
	VkRect2D sc = {{0, 0}, {W, H}};
	vkCmdSetViewport(cmd, 0, 1, &vpt);
	vkCmdSetScissor(cmd, 0, 1, &sc);
	const VkBuffer xb[2] = {x0.b, x1.b}, cb2[2] = {ctr.b, ctr.b};
	const VkDeviceSize xo[2] = {OFF0, OFF1}, xs[2] = {k->room0 * STRIDE0, k->room1 * STRIDE1};
	const VkDeviceSize co[2] = {CTR_OFF0, CTR_OFF1};
	bind_xfb(cmd, 0, 2, xb, xo, xs);
	if (k->query)
		begin_query(cmd, qp, 1, 0, 0);
	if (!k->resume)
		begin_xfb(cmd, 0, 0, NULL, NULL);
	for (unsigned d = 0; d < k->draws; d++) {
		if (k->resume)
			begin_xfb(cmd, 0, d ? 2 : 0, d ? cb2 : NULL, d ? co : NULL);
		vkCmdDraw(cmd, 6, 1, 6 * d, 0);
		if (k->resume)
			end_xfb(cmd, 0, 2, cb2, co);
	}
	if (!k->resume)
		end_xfb(cmd, 0, 2, cb2, co);
	if (k->query)
		end_query(cmd, qp, 1, 0);
	vkCmdEndRenderPass(cmd);
	if (k->query)
		vkCmdCopyQueryPoolResults(cmd, qp, 1, 1, qb.b, 8, 16, VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
	VkResult r = vkEndCommandBuffer(cmd);
	if (r != VK_SUCCESS) {
		printf(" result=FAILED (EndCommandBuffer %d: -8 refused)\n", r);
		return 4;
	}
	VkFence f;
	VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	CHECK(vkCreateFence(dev, &fi, NULL, &f));
	VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd};
	r = vkQueueSubmit(queue, 1, &si, f);
	if (r == VK_SUCCESS)
		r = vkWaitForFences(dev, 1, &f, VK_TRUE, 30000000000ull);
	vkDestroyFence(dev, f, NULL);
	if (r != VK_SUCCESS) {
		printf(" result=FAILED (VkResult %d: -8 refused, -4 device lost, 2 fence timeout)\n", r);
		return 4;
	}
	int res = 0;
	if (offline) {
		/* the query's counts are the driver's CPU work, real offline too */
		if (k->query) {
			uint64_t got[3] = {0};
			uint32_t c32[3];
			VkResult qr = vkGetQueryPoolResults(dev, qp, 1, 1, sizeof(got), got, sizeof(got),
							    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
			memcpy(c32, qb.map + 8, sizeof(c32));
			printf(" query=%d written=%llu needed=%llu available=%llu copied=%u,%u,%u", qr, (unsigned long long)got[0],
			       (unsigned long long)got[1], (unsigned long long)got[2], c32[0], c32[1], c32[2]);
		}
		printf(" result=ACCEPTED\n");
	} else {
		/* the capture model: whole triangles, in stream order, while both buffers have room */
		uint8_t want0[BUF_BYTES], want1[BUF_BYTES], wantc[64];
		memset(want0, 0xa5, sizeof(want0));
		memset(want1, 0xa5, sizeof(want1));
		memset(wantc, 0xa5, sizeof(wantc));
		uint32_t n = 0; /* records written */
		for (unsigned d = 0; d < k->draws; d++)
			for (unsigned tri = 0; tri < 2; tri++) {
				if (n + 3 > k->room0 || n + 3 > k->room1)
					continue;
				for (unsigned i = 0; i < 3; i++, n++) {
					const unsigned vtx = 3 * tri + i;
					const float *p = pos[d][vtx];
					const float v = (float)(6 * d + vtx);
					const float a[4] = {2 * p[0] + v, 2 * p[1] + v, 2 * p[2] + v, 2 * p[3] + v};
					const float b[4] = {v, p[0], 1.0f, -p[1]};
					memcpy(want0 + OFF0 + n * STRIDE0, a, 16);
					memcpy(want1 + OFF1 + n * STRIDE1, b, 16);
				}
			}
		const uint32_t c0 = n * STRIDE0, c1 = n * STRIDE1;
		memcpy(wantc + CTR_OFF0, &c0, 4);
		memcpy(wantc + CTR_OFF1, &c1, 4);
		bool first = true;
		const uint64_t b0 = diff("buf0", x0.map, want0, sizeof(want0), &first);
		const uint64_t b1 = diff("buf1", x1.map, want1, sizeof(want1), &first);
		const uint64_t bc = diff("counters", ctr.map, wantc, sizeof(wantc), &first);
		/* the stream query: primitives written (those that fit) and needed (every captured one) */
		uint64_t bq = 0;
		if (k->query) {
			const uint64_t written = n / 3, needed = 2 * k->draws;
			uint64_t got[3] = {0};
			VkResult qr = vkGetQueryPoolResults(dev, qp, 1, 1, sizeof(got), got, sizeof(got),
							    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
			uint8_t wantq[64];
			memset(wantq, 0xa5, sizeof(wantq));
			const uint32_t w32[3] = {(uint32_t)written, (uint32_t)needed, 1};
			memcpy(wantq + 8, w32, sizeof(w32));
			bq = diff("copied query", qb.map, wantq, sizeof(wantq), &first);
			if (qr != VK_SUCCESS || got[0] != written || got[1] != needed || got[2] != 1) {
				bq++;
				if (first) {
					printf(" first=query got=%d:%llu,%llu,%llu want=%llu,%llu,1", qr, (unsigned long long)got[0],
					       (unsigned long long)got[1], (unsigned long long)got[2],
					       (unsigned long long)written, (unsigned long long)needed);
					first = false;
				}
			}
			printf(" written=%llu needed=%llu", (unsigned long long)got[0], (unsigned long long)got[1]);
		}
		uint64_t bad = 0;
		for (uint32_t y = 0; y < H; y++)
			for (uint32_t x = 0; x < W; x++) {
				uint32_t want = bgra(clear_c);
				for (unsigned d = 0; d < k->draws && !k->discard; d++) {
					const struct quad *q = &k->q[d];
					if (x >= q->x0 && x < q->x1 && y >= q->y0 && y < q->y1)
						want = bgra(quad_c);
				}
				uint32_t got;
				memcpy(&got, t.map + t.img_offset + y * t.row_pitch + 4 * x, 4);
				if (got != want && !bad++ && first) {
					printf(" first=pixel%u,%u got=%08x want=%08x", x, y, got, want);
					first = false;
				}
			}
		const uint64_t gb = guard_bad(&t);
		printf(" records=%u pixels=%llu buf0=%llu buf1=%llu counters=%llu query=%llu guard=%llu", n,
		       (unsigned long long)bad, (unsigned long long)b0, (unsigned long long)b1, (unsigned long long)bc,
		       (unsigned long long)bq, (unsigned long long)gb);
		res = gb ? 2 : bad || b0 || b1 || bc || bq ? 1 : 0;
		printf(" result=%s\n", res == 2 ? "GUARD" : res ? "MISMATCH" : "EXACT");
	}
	vkFreeCommandBuffers(dev, pool, 1, &cmd);
	vkDestroyPipeline(dev, pipe, NULL);
	vkDestroyFramebuffer(dev, fb, NULL);
	vkDestroyRenderPass(dev, rp, NULL);
	vkDestroyPipelineLayout(dev, layout, NULL);
	free_buf(&vb);
	free_buf(&x0);
	free_buf(&x1);
	free_buf(&ctr);
	if (k->query) {
		vkDestroyQueryPool(dev, qp, NULL);
		free_buf(&qb);
	}
	vkDestroyImageView(dev, t.view, NULL);
	vkDestroyImage(dev, t.img, NULL);
	vkFreeMemory(dev, t.mem, NULL);
	return res;
}

int main(int argc, char **argv)
{
	const char *only = NULL;
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--offline"))
			offline = true;
		else if (!strcmp(argv[i], "--case") && i + 1 < argc)
			only = argv[++i];
		else {
			fprintf(stderr, "usage: %s [--offline] [--case NAME[,NAME...]]\n", argv[0]);
			return 3;
		}
	}
	setvbuf(stdout, NULL, _IOLBF, 0);
	VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "xfb-vk", .apiVersion = VK_API_VERSION_1_1};
	VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
	VkInstance inst;
	CHECK(vkCreateInstance(&ici, NULL, &inst));
	uint32_t n = 1;
	VkPhysicalDevice pd;
	VkResult er = vkEnumeratePhysicalDevices(inst, &n, &pd);
	if ((er != VK_SUCCESS && er != VK_INCOMPLETE) || !n) {
		fprintf(stderr, "xfb-vk: no physical device (%d)\n", er);
		return 3;
	}
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(pd, &props);
	vkGetPhysicalDeviceMemoryProperties(pd, &memprops);
	VkPhysicalDeviceTransformFeedbackFeaturesEXT tff = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT};
	VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &tff};
	vkGetPhysicalDeviceFeatures2(pd, &f2);
	VkPhysicalDeviceTransformFeedbackPropertiesEXT tfp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_PROPERTIES_EXT};
	VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &tfp};
	vkGetPhysicalDeviceProperties2(pd, &p2);
	printf("device=%s offline=%d transformFeedback=%u buffers=%u streams=%u queries=%u draw=%u\n", props.deviceName,
	       offline, tff.transformFeedback, tfp.maxTransformFeedbackBuffers, tfp.maxTransformFeedbackStreams,
	       tfp.transformFeedbackQueries, tfp.transformFeedbackDraw);
	if (!tff.transformFeedback || tfp.maxTransformFeedbackBuffers < 2) {
		fprintf(stderr, "xfb-vk: no transform feedback with two buffers\n");
		return 3;
	}
	VkPhysicalDeviceTransformFeedbackFeaturesEXT en = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT,
							   .transformFeedback = VK_TRUE};
	const char *ext = VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME;
	float prio = 1.0f;
	VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1, .pQueuePriorities = &prio};
	VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &en, .queueCreateInfoCount = 1,
				  .pQueueCreateInfos = &qci, .enabledExtensionCount = 1, .ppEnabledExtensionNames = &ext};
	CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
	bind_xfb = (PFN_vkCmdBindTransformFeedbackBuffersEXT)vkGetDeviceProcAddr(dev, "vkCmdBindTransformFeedbackBuffersEXT");
	begin_xfb = (PFN_vkCmdBeginTransformFeedbackEXT)vkGetDeviceProcAddr(dev, "vkCmdBeginTransformFeedbackEXT");
	end_xfb = (PFN_vkCmdEndTransformFeedbackEXT)vkGetDeviceProcAddr(dev, "vkCmdEndTransformFeedbackEXT");
	begin_query = (PFN_vkCmdBeginQueryIndexedEXT)vkGetDeviceProcAddr(dev, "vkCmdBeginQueryIndexedEXT");
	end_query = (PFN_vkCmdEndQueryIndexedEXT)vkGetDeviceProcAddr(dev, "vkCmdEndQueryIndexedEXT");
	if (!bind_xfb || !begin_xfb || !end_xfb || !begin_query || !end_query) {
		fprintf(stderr, "xfb-vk: no transform feedback entry points\n");
		return 3;
	}
	vkGetDeviceQueue(dev, 0, 0, &queue);
	VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
				       .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
	CHECK(vkCreateCommandPool(dev, &pci, NULL, &pool));
	int worst = 0, ran = 0;
	char list[256];
	snprintf(list, sizeof(list), "%s", only ? only : "rec,two,overflow,resume,query,discard");
	for (char *save = NULL, *name = strtok_r(list, ",", &save); name; name = strtok_r(NULL, ",", &save)) {
		const struct tcase *k = NULL;
		for (unsigned c = 0; c < sizeof(cases) / sizeof(cases[0]); c++)
			if (!strcmp(name, cases[c].name))
				k = &cases[c];
		if (!k)
			return fprintf(stderr, "xfb-vk: no case %s\n", name), 3;
		int r = run_case(k);
		ran++;
		if (r == 4) {
			printf("result=REFUSED\n");
			return 4;
		}
		worst = r > worst ? r : worst;
	}
	if (!ran)
		return fprintf(stderr, "xfb-vk: no case\n"), 3;
	vkDestroyCommandPool(dev, pool, NULL);
	vkDestroyDevice(dev, NULL);
	vkDestroyInstance(inst, NULL);
	printf("result=%s\n", offline ? "PASS-OFFLINE" : worst == 0 ? "PASS" : worst == 2 ? "GUARD" : "MISMATCH");
	return offline ? 0 : worst;
}
