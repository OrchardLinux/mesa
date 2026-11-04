/* SPDX-License-Identifier: MIT */
/* xfb-test: transform feedback (VK_EXT_transform_feedback), checked byte for byte against a CPU model.
 * Ported from the Apple A9 (G5P) Linux port's fixture. The A9-only diagnostics (--diag, --diag2: register-hold
 * and store-width probes) and the vertex-input ladder (--vi=: the A9 vertex PDS) are not in this copy.
 *
 * Every case renders into a 64x64 B8G8R8A8 target (the jobs need one) and captures into a buffer prefilled with
 * 0xa5. The binding starts 32 bytes into the buffer, so stray bytes before the binding, after the last record and
 * in stride padding are all visible. Records are (id, ~id, float(id), 1.0) per vertex unless noted.
 *
 * Suite (default; --case=NAME runs one):
 *   rec        xfb-rec.vert, 6 vertices (2 triangles), no attributes: 6 records
 *   first3     the same with firstVertex 3: records 3..8 (gl_VertexIndex includes the first vertex)
 *   overflow   a 48-byte binding: 1 triangle (3 records) fits, counter 48; with transformFeedbackQueries also
 *              the stream query: written 1 / needed 2
 *   instanced  xfb-inst.vert, 3 vertices x 2 instances: instance-major records (id, instance, id + 7, 1.0)
 *   attr       xfb-attr.vert: a fetched position + the record; buffer 1 captures the position at stride 32
 *   attr16     (only with --case=attr16) the same with buffer 1 at stride 16
 *   discard    two triangles covering the target with rasterizer discard (dynamic, EDS2): records and positions
 *              are captured and the target keeps its clear colour; the same draw without discard is red
 *   resume     two render passes, End/Begin through a counter buffer; with transformFeedbackDraw a third pass
 *              draws vkCmdDrawIndirectByteCountEXT from it (skipped, and said so, without it)
 *   dup        3000 vertices captured into a buffer that is also the vertex buffer (xfb-dup.vert): exactly one
 *              exact record per vertex, nothing past the binding (once = shaded once, twice/more = shaded again
 *              after its record landed; both are correct, it is knowledge)
 * Each prints one case= line with result=PASS|FAIL, then result=.
 *
 * --run3: record (6 records of the attribute-less cover VS xfb-cover.vert, which also covers the target), overflow
 * (48-byte binding: 3 records, counter 48), discard (the record draw with rasterizer discard: no pixel, the same
 * records), dup (as above). One r3= line each (lanes= counts bad bytes per dword of the records, i.e. per 4-byte
 * store), then r3-result.
 *
 * --ctl, --dup=N[,sep][,nocap]: one job per process. ctl is run 3's record job alone; dup=N draws xfb-dup.vert over
 * N vertices (N a multiple of 3), capturing into a buffer that is also the vertex buffer, or with sep a separate
 * zero-filled vertex buffer, or with nocap without capture (the capture memory must stay untouched). One dl= line.
 *
 * --offline (a driver that executes nothing): only what the CPU computes is checked (counters, queries, the
 * byte-count vertex count, every submit accepted). Exit 0 pass, 1 a check failed, 3 setup error.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "xfb-rec.vert.h"
#include "xfb-attr.vert.h"
#include "xfb-attr16.vert.h"
#include "xfb-inst.vert.h"
#include "xfb-dup.vert.h"
#include "xfb-cover.vert.h"
#include "xfb.frag.h"

#define CHECK(x)                                                                         \
	do {                                                                             \
		VkResult r_ = (x);                                                       \
		if (r_ != VK_SUCCESS) {                                                  \
			fprintf(stderr, "xfb-test: %s = %d (line %d)\n", #x, r_, __LINE__); \
			exit(3);                                                         \
		}                                                                        \
	} while (0)

#define W 64
#define GUARD 0xa5
#define BIND_OFFSET 32
#define MIN_FIT(a, b) ((a) < (b) ? (a) : (b))

static VkDevice dev;
static VkPhysicalDevice pd;
static VkQueue queue;
static VkCommandPool pool;
static bool offline;
static int failed;
static bool have_queries, have_draw; /* VkPhysicalDeviceTransformFeedbackPropertiesEXT */
static PFN_vkCmdBindTransformFeedbackBuffersEXT bind_xfb;
static PFN_vkCmdBeginTransformFeedbackEXT begin_xfb;
static PFN_vkCmdEndTransformFeedbackEXT end_xfb;
static PFN_vkCmdDrawIndirectByteCountEXT draw_byte_count;
static PFN_vkCmdBeginQueryIndexedEXT begin_query;
static PFN_vkCmdEndQueryIndexedEXT end_query;
static PFN_vkCmdSetRasterizerDiscardEnableEXT set_discard; /* NULL without VK_EXT_extended_dynamic_state2 */
static PFN_vkCmdBeginRenderingKHR begin_rendering;
static PFN_vkCmdEndRenderingKHR end_rendering;

struct buf {
	VkBuffer b;
	VkDeviceMemory m;
	uint8_t *map;
	VkDeviceSize size;
};

static uint32_t mem_type(uint32_t bits)
{
	VkPhysicalDeviceMemoryProperties mp;
	vkGetPhysicalDeviceMemoryProperties(pd, &mp);
	for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
		if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
			return i;
	return 0;
}

static struct buf buffer(VkDeviceSize size, VkBufferUsageFlags usage, uint8_t fill)
{
	struct buf r = {.size = size};
	VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage};
	CHECK(vkCreateBuffer(dev, &bi, NULL, &r.b));
	VkMemoryRequirements mr;
	vkGetBufferMemoryRequirements(dev, r.b, &mr);
	VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
				   .memoryTypeIndex = mem_type(mr.memoryTypeBits)};
	CHECK(vkAllocateMemory(dev, &ai, NULL, &r.m));
	CHECK(vkBindBufferMemory(dev, r.b, r.m, 0));
	CHECK(vkMapMemory(dev, r.m, 0, VK_WHOLE_SIZE, 0, (void **)&r.map));
	memset(r.map, fill, size);
	return r;
}

static void buffer_free(struct buf *b)
{
	vkUnmapMemory(dev, b->m);
	vkDestroyBuffer(dev, b->b, NULL);
	vkFreeMemory(dev, b->m, NULL);
}

struct target {
	VkImage img;
	VkDeviceMemory m;
	VkImageView view;
};

static struct target target(void)
{
	struct target t;
	VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
				.format = VK_FORMAT_B8G8R8A8_UNORM, .extent = {W, W, 1}, .mipLevels = 1, .arrayLayers = 1,
				.samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_LINEAR,
				.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
	CHECK(vkCreateImage(dev, &ii, NULL, &t.img));
	VkMemoryRequirements mr;
	vkGetImageMemoryRequirements(dev, t.img, &mr);
	VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
				   .memoryTypeIndex = mem_type(mr.memoryTypeBits)};
	CHECK(vkAllocateMemory(dev, &ai, NULL, &t.m));
	CHECK(vkBindImageMemory(dev, t.img, t.m, 0));
	VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = t.img,
				    .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_B8G8R8A8_UNORM,
				    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
	CHECK(vkCreateImageView(dev, &vi, NULL, &t.view));
	return t;
}

static VkShaderModule module(const uint32_t *code, size_t bytes)
{
	VkShaderModuleCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = bytes, .pCode = code};
	VkShaderModule m;
	CHECK(vkCreateShaderModule(dev, &ci, NULL, &m));
	return m;
}

static VkPipelineLayout layout;
static bool may_refuse; /* pipeline(): VK_ERROR_FEATURE_NOT_PRESENT gives VK_NULL_HANDLE instead of exit */

/* attribs: one RGBA32F attribute at location 0, binding 0, stride 16 */
static VkPipeline pipeline(const uint32_t *vs, size_t vs_bytes, bool attribs, bool dynamic_discard)
{
	VkVertexInputBindingDescription bd = {0, 16, VK_VERTEX_INPUT_RATE_VERTEX};
	VkVertexInputAttributeDescription ad = {0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0};
	VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
						   .vertexBindingDescriptionCount = attribs, .pVertexBindingDescriptions = &bd,
						   .vertexAttributeDescriptionCount = attribs, .pVertexAttributeDescriptions = &ad};
	VkShaderModule v = module(vs, vs_bytes), f = module(xfb_frag, sizeof(xfb_frag));
	VkPipelineShaderStageCreateInfo st[2] = {
		{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = v, .pName = "main"},
		{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = f, .pName = "main"},
	};
	VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
						     .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
	VkViewport vp = {0, 0, W, W, 0, 1};
	VkRect2D sc = {{0, 0}, {W, W}};
	VkPipelineViewportStateCreateInfo vs_ = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
						 .viewportCount = 1, .pViewports = &vp, .scissorCount = 1, .pScissors = &sc};
	VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
						     .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
						     .lineWidth = 1.0f};
	VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
						   .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
	VkPipelineColorBlendAttachmentState ba = {.colorWriteMask = 0xf};
	VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
						  .attachmentCount = 1, .pAttachments = &ba};
	VkDynamicState ds = VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE;
	VkPipelineDynamicStateCreateInfo dyn = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
						.dynamicStateCount = dynamic_discard && set_discard, .pDynamicStates = &ds};
	VkFormat fmt = VK_FORMAT_B8G8R8A8_UNORM;
	VkPipelineRenderingCreateInfoKHR ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
					       .colorAttachmentCount = 1, .pColorAttachmentFormats = &fmt};
	VkGraphicsPipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri,
					   .stageCount = 2, .pStages = st, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
					   .pViewportState = &vs_, .pRasterizationState = &rs, .pMultisampleState = &ms,
					   .pColorBlendState = &cb, .pDynamicState = &dyn, .layout = layout};
	VkPipeline p = VK_NULL_HANDLE;
	VkResult r = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &ci, NULL, &p);
	if (r == VK_ERROR_FEATURE_NOT_PRESENT && may_refuse)
		p = VK_NULL_HANDLE;
	else
		CHECK(r);
	vkDestroyShaderModule(dev, v, NULL);
	vkDestroyShaderModule(dev, f, NULL);
	return p;
}

static VkCommandBuffer begin_cb(void)
{
	VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
					  .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
	VkCommandBuffer cb;
	CHECK(vkAllocateCommandBuffers(dev, &ai, &cb));
	VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
				       .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
	CHECK(vkBeginCommandBuffer(cb, &bi));
	return cb;
}

static void submit(VkCommandBuffer cb)
{
	CHECK(vkEndCommandBuffer(cb));
	VkFence f;
	VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	CHECK(vkCreateFence(dev, &fi, NULL, &f));
	VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
	CHECK(vkQueueSubmit(queue, 1, &si, f));
	CHECK(vkWaitForFences(dev, 1, &f, VK_TRUE, 60000000000ull));
	vkDestroyFence(dev, f, NULL);
	vkFreeCommandBuffers(dev, pool, 1, &cb);
}

static void render(VkCommandBuffer cb, const struct target *t)
{
	VkRenderingAttachmentInfoKHR ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR, .imageView = t->view,
					   .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
					   .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
					   .clearValue.color.float32 = {0.0f, 0.0f, 1.0f, 1.0f}};
	VkRenderingInfoKHR ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR, .renderArea = {{0, 0}, {W, W}},
				 .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &ca};
	begin_rendering(cb, &ri);
}

/* (id, ~id, float(id), 1.0) or, instanced, (id, instance, id + 7, 1.0) */
static void record(uint8_t out[16], uint32_t v, uint32_t inst, bool instanced)
{
	float fv = (float)v, one = 1.0f;
	uint32_t w[4] = {v, instanced ? inst : ~v, instanced ? v + 7 : 0, 0};
	if (!instanced)
		memcpy(&w[2], &fv, 4);
	memcpy(&w[3], &one, 4);
	memcpy(out, w, 16);
}

/* Per primitive: exact records, the counter, untouched guard bytes. pos: the expected captured positions (one
 * vec4 per vertex), NULL = records. */
static int check_capture(const char *name, const struct buf *b, uint32_t counter, uint32_t stride, uint32_t rec_bytes,
			 uint32_t vertices, uint32_t instances, uint32_t first, uint32_t want_records, const float (*pos)[4])
{
	if (counter != want_records * stride) {
		printf("case=%s counter=%u want=%u result=FAIL\n", name, counter, want_records * stride);
		return 1;
	}
	if (offline)
		return 0;
	unsigned bad = 0, first_bad = ~0u;
	uint32_t complete = vertices / 3 * 3; /* each instance's complete triangles only */
	for (uint32_t k = 0; k < b->size; k++) {
		uint8_t want = GUARD;
		if (k >= BIND_OFFSET) {
			uint32_t rel = k - BIND_OFFSET, r = rel / stride, in = rel % stride;
			if (r < want_records && in < rec_bytes) {
				uint8_t rec[16];
				if (pos)
					memcpy(rec, pos[r % complete], 16);
				else
					record(rec, first + r % complete, r / complete, instances > 1);
				want = rec[in];
			}
		}
		if (b->map[k] != want && !bad++)
			first_bad = k;
	}
	if (bad)
		printf("case=%s bad_bytes=%u first_bad=%u (got %#x) result=FAIL\n", name, bad, first_bad, b->map[first_bad]);
	return bad != 0;
}

struct xcase {
	const char *name;
	uint32_t vertices, instances, first, bind_size;
	bool query;
};

static void capture_case(const struct xcase *k, VkPipeline p, const struct target *t)
{
	struct buf x = buffer(512, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT, GUARD);
	struct buf ctr = buffer(64, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT, 0);
	VkQueryPool qp = VK_NULL_HANDLE;
	if (k->query && have_queries) {
		VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
					    .queryType = VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT, .queryCount = 1};
		CHECK(vkCreateQueryPool(dev, &qi, NULL, &qp));
	}
	VkCommandBuffer cb = begin_cb();
	if (qp)
		vkCmdResetQueryPool(cb, qp, 0, 1);
	render(cb, t);
	vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
	if (set_discard)
		set_discard(cb, VK_FALSE);
	VkDeviceSize off = BIND_OFFSET, size = k->bind_size;
	bind_xfb(cb, 0, 1, &x.b, &off, &size);
	if (qp)
		begin_query(cb, qp, 0, 0, 0);
	begin_xfb(cb, 0, 0, NULL, NULL);
	vkCmdDraw(cb, k->vertices, k->instances, k->first, 0);
	VkDeviceSize zero = 0;
	end_xfb(cb, 0, 1, &ctr.b, &zero);
	if (qp)
		end_query(cb, qp, 0, 0);
	end_rendering(cb);
	submit(cb);
	uint32_t counter;
	memcpy(&counter, ctr.map, 4);
	uint32_t prims = k->vertices / 3 * k->instances, fit = MIN_FIT(k->bind_size / 48, prims);
	int bad = check_capture(k->name, &x, counter, 16, 16, k->vertices, k->instances, k->first, fit * 3, NULL);
	if (qp) {
		uint64_t q[2] = {0};
		VkResult r = vkGetQueryPoolResults(dev, qp, 0, 1, sizeof(q), q, sizeof(q),
						   VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
		if (r != VK_SUCCESS || q[0] != fit || q[1] != prims) {
			printf("case=%s query written=%llu needed=%llu want %u/%u result=FAIL\n", k->name,
			       (unsigned long long)q[0], (unsigned long long)q[1], fit, prims);
			bad = 1;
		}
		vkDestroyQueryPool(dev, qp, NULL);
	}
	if (!bad)
		printf("case=%s records=%u counter=%u%s result=%s\n", k->name, fit * 3, counter,
		       k->query && !qp ? " query=skipped(no transformFeedbackQueries)" : "", offline ? "PASS-OFFLINE" : "PASS");
	failed |= bad;
	buffer_free(&x);
	buffer_free(&ctr);
}

/* Pixels of rows y0..y1-1 of the target that are not the clear colour (B8G8R8A8 blue: ff 00 00 ff). */
static unsigned non_clear_rows(const struct target *t, uint32_t y0, uint32_t y1)
{
	const uint8_t *im;
	CHECK(vkMapMemory(dev, t->m, 0, VK_WHOLE_SIZE, 0, (void **)&im));
	VkImageSubresource sub = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
	VkSubresourceLayout lay;
	vkGetImageSubresourceLayout(dev, t->img, &sub, &lay);
	unsigned px = 0;
	for (uint32_t y = y0; y < y1; y++)
		for (uint32_t x = 0; x < W; x++)
			px += memcmp(im + lay.offset + y * lay.rowPitch + 4 * x, "\xff\x00\x00\xff", 4) != 0;
	vkUnmapMemory(dev, t->m);
	return px;
}

static unsigned non_clear(const struct target *t)
{
	return non_clear_rows(t, 0, W);
}

/* attr: 6 thin triangles at x = -1 + 0.25 i, y = -1 + 0.5 (i % 3). discard: two triangles covering the target. */
static const float attr_pos[6][4] = {{-1.0f, -1.0f, 0, 1}, {-0.75f, -0.5f, 0, 1}, {-0.5f, 0.0f, 0, 1},
				     {-0.25f, -1.0f, 0, 1}, {0.0f, -0.5f, 0, 1}, {0.25f, 0.0f, 0, 1}};
static const float cover_pos[6][4] = {{-1, -1, 0, 1}, {1, -1, 0, 1}, {-1, 1, 0, 1}, {1, -1, 0, 1}, {1, 1, 0, 1}, {-1, 1, 0, 1}};

/* One draw of xfb-attr.vert (records into x0, positions into x1); discard < 0: no discard state set. */
static uint32_t attr_stride1 = 32; /* buffer 1's stride: xfb-attr.vert 32, xfb-attr16.vert (--case=attr16) 16 */

static int attr_draw(const char *name, VkPipeline p, const float (*pos)[4], int discard, const struct target *t)
{
	struct buf vb = buffer(6 * 16, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 0);
	memcpy(vb.map, pos, 6 * 16);
	struct buf x0 = buffer(512, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT, GUARD);
	struct buf x1 = buffer(512, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT, GUARD);
	struct buf ctr = buffer(64, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT, 0);
	VkCommandBuffer cb = begin_cb();
	render(cb, t);
	vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
	if (discard >= 0)
		set_discard(cb, discard);
	VkDeviceSize zero = 0;
	vkCmdBindVertexBuffers(cb, 0, 1, &vb.b, &zero);
	VkBuffer xb[2] = {x0.b, x1.b};
	VkDeviceSize offs[2] = {BIND_OFFSET, BIND_OFFSET}, sizes[2] = {256, 256}, coffs[2] = {0, 16};
	bind_xfb(cb, 0, 2, xb, offs, sizes);
	begin_xfb(cb, 0, 0, NULL, NULL);
	vkCmdDraw(cb, 6, 1, 0, 0);
	VkBuffer cbs[2] = {ctr.b, ctr.b};
	end_xfb(cb, 0, 2, cbs, coffs);
	end_rendering(cb);
	submit(cb);
	uint32_t c0, c1;
	memcpy(&c0, ctr.map, 4);
	memcpy(&c1, ctr.map + 16, 4);
	char n0[64], n1[64];
	snprintf(n0, sizeof(n0), "%s-rec", name);
	snprintf(n1, sizeof(n1), "%s-pos", name);
	int bad = check_capture(n0, &x0, c0, 16, 16, 6, 1, 0, 6, NULL);
	bad |= check_capture(n1, &x1, c1, attr_stride1, 16, 6, 1, 0, 6, pos);
	if (!bad)
		printf("case=%s records=6 counters=%u/%u result=%s\n", name, c0, c1, offline ? "PASS-OFFLINE" : "PASS");
	buffer_free(&vb);
	buffer_free(&x0);
	buffer_free(&x1);
	buffer_free(&ctr);
	return bad;
}

/* xfb-attr.vert with two capture buffers. The A9 driver refused it at pipeline creation until its vertex temps
 * limit was lifted; a refusal (VK_ERROR_FEATURE_NOT_PRESENT) is reported as REFUSED, not a crash. */
static VkPipeline attr_pipeline(const char *name, const uint32_t *vs, size_t vs_bytes, bool dynamic_discard)
{
	may_refuse = true;
	VkPipeline p = pipeline(vs, vs_bytes, true, dynamic_discard);
	may_refuse = false;
	if (!p) {
		printf("case=%s result=REFUSED (pipeline creation: VK_ERROR_FEATURE_NOT_PRESENT)\n", name);
		failed = 1;
	}
	return p;
}

/* xfb-attr.vert with buffer 1 at stride 16, so both buffers use the record's offset (the second buffer's address
 * add and stores alone). */
static void attr16_case(const struct target *t)
{
	VkPipeline p = attr_pipeline("attr16", xfb_attr16_vert, sizeof(xfb_attr16_vert), false);
	if (!p)
		return;
	attr_stride1 = 16;
	failed |= attr_draw("attr16", p, attr_pos, -1, t);
	attr_stride1 = 32;
	vkDestroyPipeline(dev, p, NULL);
}

static void attr_case(const struct target *t)
{
	VkPipeline p = attr_pipeline("attr", xfb_attr_vert, sizeof(xfb_attr_vert), false);
	if (!p)
		return;
	failed |= attr_draw("attr", p, attr_pos, -1, t);
	vkDestroyPipeline(dev, p, NULL);
}

/* Rasterizer discard: the capture still happens and no pixel changes; the control draw (no discard) covers the target. */
static void discard_case(const struct target *t)
{
	if (!set_discard) {
		printf("case=discard result=SKIP (no VK_EXT_extended_dynamic_state2)\n");
		return;
	}
	VkPipeline p = attr_pipeline("discard", xfb_attr_vert, sizeof(xfb_attr_vert), true);
	if (!p)
		return;
	int bad = attr_draw("discard", p, cover_pos, 1, t);
	unsigned on = offline ? 0 : non_clear(t);
	int ctl = attr_draw("discard-control", p, cover_pos, 0, t);
	unsigned off = offline ? W * W : non_clear(t);
	if (on || off < W * W / 2) {
		printf("case=discard non_clear_with_discard=%u non_clear_without=%u (want 0 and the whole target) result=FAIL\n", on, off);
		bad = 1;
	}
	printf("case=discard pixels_with=%u pixels_without=%u control=%s result=%s\n", on, off, ctl ? "FAIL" : "ok",
	       bad ? "FAIL" : offline ? "PASS-OFFLINE" : "PASS");
	failed |= bad | ctl;
	vkDestroyPipeline(dev, p, NULL);
}

/* Pass 1 captures vertices 0..2 and ends into a counter; pass 2 resumes from it with vertices 3..5; pass 3 (only
 * with transformFeedbackDraw) draws the captured byte count (6 vertices, recaptured into a second buffer). */
static void resume_case(VkPipeline p, const struct target *t)
{
	struct buf x = buffer(512, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT, GUARD);
	struct buf y = buffer(512, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT, GUARD);
	struct buf ctr = buffer(64, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT |
					   VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, 0);
	VkCommandBuffer cb = begin_cb();
	VkDeviceSize off = BIND_OFFSET, size = 256, zero = 0, c2 = 32;
	unsigned passes = have_draw ? 3 : 2;
	for (unsigned pass = 0; pass < passes; pass++) {
		render(cb, t);
		vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
		if (set_discard)
			set_discard(cb, VK_FALSE);
		bind_xfb(cb, 0, 1, pass < 2 ? &x.b : &y.b, &off, &size);
		begin_xfb(cb, 0, pass == 1, &ctr.b, &zero);
		if (pass < 2)
			vkCmdDraw(cb, 3, 1, 3 * pass, 0);
		else
			draw_byte_count(cb, 1, 0, ctr.b, 0, 0, 16);
		end_xfb(cb, 0, 1, &ctr.b, pass < 2 ? &zero : &c2);
		end_rendering(cb);
	}
	submit(cb);
	uint32_t c, cy = 0;
	memcpy(&c, ctr.map, 4);
	memcpy(&cy, ctr.map + 32, 4);
	int bad = check_capture("resume", &x, c, 16, 16, 6, 1, 0, 6, NULL);
	if (have_draw)
		bad |= check_capture("bytecount", &y, cy, 16, 16, 6, 1, 0, 6, NULL);
	if (!bad)
		printf("case=resume counter=%u bytecount_counter=%u%s result=%s\n", c, cy,
		       have_draw ? "" : " bytecount=skipped(no transformFeedbackDraw)", offline ? "PASS-OFFLINE" : "PASS");
	failed |= bad;
	buffer_free(&x);
	buffer_free(&y);
	buffer_free(&ctr);
}

/* ---- --run3 ---- */
/* One captured draw of the attribute-less cover VS (xfb-cover.vert: vertices 0..2 cover the target, 3..5 lie off
 * it): `vertices` records into a GUARD-filled buffer bound at BIND_OFFSET with `bind` bytes, with a stream query
 * (when the device has them) and dynamic rasterizer discard. Exact readback plus the bad bytes per dword lane of the
 * records: lane k is the dword at byte offset 4k of each record, i.e. one 4-byte store. */
static int run3_capture(const char *name, VkPipeline p, uint32_t vertices, uint32_t bind, int discard,
			const struct target *t)
{
	struct buf x = buffer(512, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT, GUARD);
	struct buf ctr = buffer(64, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT, 0);
	VkQueryPool qp = VK_NULL_HANDLE;
	if (have_queries) {
		VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
					    .queryType = VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT, .queryCount = 1};
		CHECK(vkCreateQueryPool(dev, &qi, NULL, &qp));
	}
	VkCommandBuffer cb = begin_cb();
	if (qp)
		vkCmdResetQueryPool(cb, qp, 0, 1);
	render(cb, t);
	vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
	set_discard(cb, discard);
	VkDeviceSize off = BIND_OFFSET, size = bind, zero = 0;
	bind_xfb(cb, 0, 1, &x.b, &off, &size);
	if (qp)
		begin_query(cb, qp, 0, 0, 0);
	begin_xfb(cb, 0, 0, NULL, NULL);
	vkCmdDraw(cb, vertices, 1, 0, 0);
	end_xfb(cb, 0, 1, &ctr.b, &zero);
	if (qp)
		end_query(cb, qp, 0, 0);
	end_rendering(cb);
	submit(cb);
	unsigned px = offline ? (discard ? 0 : W * W) : non_clear(t);
	uint32_t counter, prims = vertices / 3, fit = MIN_FIT(bind / 48, prims), records = fit * 3;
	memcpy(&counter, ctr.map, 4);
	int bad = check_capture(name, &x, counter, 16, 16, vertices, 1, 0, records, NULL);
	unsigned lane[4] = {0}, outside = 0;
	for (uint32_t k = 0; k < x.size && !offline; k++) {
		uint32_t rel = k - BIND_OFFSET;
		bool in = k >= BIND_OFFSET && rel / 16 < records;
		uint8_t want = GUARD, rec[16];
		if (in) {
			record(rec, rel / 16, 0, false);
			want = rec[rel % 16];
		}
		if (x.map[k] != want) {
			if (in)
				lane[rel % 16 / 4]++;
			else
				outside++;
		}
	}
	uint64_t q[2] = {0};
	bool qbad = false;
	if (qp) {
		VkResult qr = vkGetQueryPoolResults(dev, qp, 0, 1, sizeof(q), q, sizeof(q),
						    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
		qbad = qr != VK_SUCCESS || q[0] != fit || q[1] != prims;
		vkDestroyQueryPool(dev, qp, NULL);
	}
	bool pxbad = discard ? px != 0 : px != W * W;
	bad |= qbad || pxbad;
	printf("r3=%s pixels=%u/%u records=%u counter=%u/%u lanes=%u/%u/%u/%u outside=%u ", name, px,
	       discard ? 0 : W * W, records, counter, records * 16, lane[0], lane[1], lane[2], lane[3], outside);
	if (qp)
		printf("query=%llu/%llu want=%u/%u ", (unsigned long long)q[0], (unsigned long long)q[1], fit, prims);
	else
		printf("query=skipped ");
	printf("result=%s\n", bad ? "FAIL" : offline ? "PASS-OFFLINE" : "PASS");
	buffer_free(&x);
	buffer_free(&ctr);
	return bad;
}

/* n vertices (n / 510 vertex-data splits on the A9: 170 triangles each); with capture, each vertex must leave exactly
 * one exact record at its own slot and nothing past the binding. alias: the capture memory is also the vertex buffer
 * (xfb-dup.vert reads it back: a vertex shaded again after its record landed gives twice/more, still correct);
 * otherwise the vertex buffer is a separate zero-filled buffer. Without capture (cap false) the capture memory must
 * stay untouched. tag "dup" prints the suite's / --run3's line (r3=dup with --run3), else dl=. */
static int dup_draw(const struct target *t, const char *tag, uint32_t n, bool alias, bool cap, const char *prefix)
{
	VkPipeline p = pipeline(xfb_dup_vert, sizeof(xfb_dup_vert), true, false);
	struct buf x = buffer((VkDeviceSize)n * 16 + 64, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT |
						  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 0);
	memset(x.map + (size_t)n * 16, GUARD, 64);
	struct buf vb = {0};
	if (!alias)
		vb = buffer((VkDeviceSize)n * 16, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 0);
	struct buf ctr = buffer(64, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT, 0);
	VkCommandBuffer cb = begin_cb();
	render(cb, t);
	vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
	VkDeviceSize zero = 0, size = (VkDeviceSize)n * 16;
	vkCmdBindVertexBuffers(cb, 0, 1, alias ? &x.b : &vb.b, &zero);
	if (cap) {
		bind_xfb(cb, 0, 1, &x.b, &zero, &size);
		begin_xfb(cb, 0, 0, NULL, NULL);
	}
	vkCmdDraw(cb, n, 1, 0, 0);
	if (cap)
		end_xfb(cb, 0, 1, &ctr.b, &zero);
	end_rendering(cb);
	submit(cb);
	uint32_t c;
	memcpy(&c, ctr.map, 4);
	unsigned hist[4] = {0}, wrong = 0, tail = 0, touched = 0;
	int first_wrong = -1;
	for (uint32_t i = 0; i < n && !offline && cap; i++) {
		float r[4];
		memcpy(r, x.map + 16 * i, 16);
		unsigned k = r[1] == 1.0f ? 1 : r[1] == 2.0f ? 2 : r[1] >= 3.0f ? 3 : 0;
		hist[k]++;
		if (r[0] != (float)i || r[2] != 0.0f || r[3] != 1.0f || !k) {
			wrong++;
			if (first_wrong < 0)
				first_wrong = (int)i;
		}
	}
	for (uint32_t k = 0; k < (size_t)n * 16 && !offline && !cap; k++)
		touched += x.map[k] != 0;
	for (uint32_t k = 0; k < 64 && !offline; k++)
		tail += x.map[(size_t)n * 16 + k] != GUARD;
	bool bad = cap ? c != n * 16 || wrong || tail : touched || tail;
	const char *res = bad ? "FAIL" : offline ? "PASS-OFFLINE" : "PASS";
	if (!strcmp(tag, "dup"))
		printf("%sdup vertices=%u counter=%u/%u exact=%u wrong=%u first_wrong=%d tail_bad=%u once=%u twice=%u "
		       "more=%u result=%s\n", prefix, n, c, n * 16, n - wrong, wrong, first_wrong, tail, hist[1], hist[2],
		       hist[3], res);
	else if (cap)
		printf("dl=%s vertices=%u vb=%s cap=1 counter=%u/%u exact=%u wrong=%u first_wrong=%d tail_bad=%u "
		       "once=%u twice=%u more=%u result=%s\n", tag, n, alias ? "alias" : "sep", c, n * 16,
		       n - wrong, wrong, first_wrong, tail, hist[1], hist[2], hist[3], res);
	else
		printf("dl=%s vertices=%u vb=%s cap=0 completed=1 touched=%u tail_bad=%u result=%s\n", tag, n,
		       alias ? "alias" : "sep", touched, tail, res);
	if (!alias)
		buffer_free(&vb);
	buffer_free(&x);
	buffer_free(&ctr);
	vkDestroyPipeline(dev, p, NULL);
	return bad;
}

static void run3(const struct target *t)
{
	if (!set_discard) {
		fprintf(stderr, "xfb-test: --run3 needs VK_EXT_extended_dynamic_state2 (rasterizer discard)\n");
		exit(3);
	}
	VkPipeline p = pipeline(xfb_cover_vert, sizeof(xfb_cover_vert), false, true);
	int rec = run3_capture("record", p, 6, 256, 0, t);
	int ovf = run3_capture("overflow", p, 6, 48, 0, t);
	int dis = run3_capture("discard", p, 6, 256, 1, t);
	vkDestroyPipeline(dev, p, NULL);
	int dup = dup_draw(t, "dup", 3000, true, true, "r3=");
	failed |= rec | ovf | dis | dup;
	printf("r3-result record=%s overflow=%s discard=%s dup=%s result=%s\n", rec ? "FAIL" : "PASS",
	       ovf ? "FAIL" : "PASS", dis ? "FAIL" : "PASS", dup ? "FAIL" : "PASS", failed ? "FAIL" : "PASS");
}

/* The dup ladder: one render job per process. --ctl repeats --run3's record job (the GPU still works after a
 * failed rung); --dup=N[,sep][,nocap] is dup_draw with N vertices, a separate vertex buffer (sep; default: the
 * capture buffer) and no capture (nocap). */
static void ladder(const struct target *t, bool ctl, const char *spec)
{
	if (ctl) {
		if (!set_discard) {
			fprintf(stderr, "xfb-test: --ctl needs VK_EXT_extended_dynamic_state2 (rasterizer discard)\n");
			exit(3);
		}
		VkPipeline p = pipeline(xfb_cover_vert, sizeof(xfb_cover_vert), false, true);
		int bad = run3_capture("record", p, 6, 256, 0, t);
		vkDestroyPipeline(dev, p, NULL);
		failed |= bad;
		printf("dl=ctl result=%s\n", bad ? "FAIL" : offline ? "PASS-OFFLINE" : "PASS");
		return;
	}
	char *end;
	unsigned long n = strtoul(spec, &end, 10);
	bool sep = strstr(end, ",sep") != NULL, nocap = strstr(end, ",nocap") != NULL;
	if (!n || n % 3 || n > 30000) {
		fprintf(stderr, "xfb-test: --dup=N[,sep][,nocap]: N a multiple of 3, 3..30000 (got %s)\n", spec);
		exit(3);
	}
	failed |= dup_draw(t, spec, (uint32_t)n, !sep, !nocap, "");
}

static bool has_ext(const VkExtensionProperties *e, uint32_t n, const char *name)
{
	for (uint32_t i = 0; i < n; i++)
		if (!strcmp(e[i].extensionName, name))
			return true;
	return false;
}

#define PROC(var, name) var = (void *)vkGetDeviceProcAddr(dev, name)

int main(int argc, char **argv)
{
	const char *only = NULL, *dup_spec = NULL;
	bool run3_mode = false, ctl_mode = false;
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--offline"))
			offline = true;
		else if (!strcmp(argv[i], "--run3"))
			run3_mode = true;
		else if (!strcmp(argv[i], "--ctl"))
			ctl_mode = true;
		else if (!strncmp(argv[i], "--dup=", 6))
			dup_spec = argv[i] + 6;
		else if (!strncmp(argv[i], "--case=", 7))
			only = argv[i] + 7;
		else {
			fprintf(stderr, "usage: %s [--offline] [--case=NAME | --run3 | --ctl | --dup=N[,sep][,nocap]]\n",
				argv[0]);
			return 3;
		}
	}
	setvbuf(stdout, NULL, _IOLBF, 0);
	VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "xfb-test",
				 .apiVersion = VK_API_VERSION_1_1};
	VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
	VkInstance inst;
	CHECK(vkCreateInstance(&ici, NULL, &inst));
	uint32_t n = 1;
	VkResult er = vkEnumeratePhysicalDevices(inst, &n, &pd);
	if ((er != VK_SUCCESS && er != VK_INCOMPLETE) || !n) {
		fprintf(stderr, "xfb-test: no physical device (%d)\n", er);
		return 3;
	}
	VkPhysicalDeviceTransformFeedbackPropertiesEXT tfp = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_PROPERTIES_EXT};
	VkPhysicalDeviceProperties2 props = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &tfp};
	vkGetPhysicalDeviceProperties2(pd, &props);
	have_queries = tfp.transformFeedbackQueries;
	have_draw = tfp.transformFeedbackDraw;

	uint32_t ne = 0;
	vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, NULL);
	VkExtensionProperties *exts = calloc(ne ? ne : 1, sizeof(*exts));
	vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, exts);
	bool eds2 = has_ext(exts, ne, "VK_EXT_extended_dynamic_state2");
	printf("device=%s offline=%d maxTransformFeedbackBuffers=%u queries=%u draw=%u eds2=%u\n",
	       props.properties.deviceName, offline, tfp.maxTransformFeedbackBuffers, have_queries, have_draw, eds2);
	static const char *const need[] = {"VK_KHR_dynamic_rendering", "VK_KHR_create_renderpass2",
					   "VK_KHR_depth_stencil_resolve", "VK_KHR_multiview", "VK_KHR_maintenance2",
					   "VK_EXT_transform_feedback"};
	const char *ext[8];
	uint32_t next = 0;
	for (unsigned i = 0; i < sizeof(need) / sizeof(need[0]); i++) {
		if (!has_ext(exts, ne, need[i])) {
			printf("result=SETUP (device lacks %s)\n", need[i]);
			return 3;
		}
		ext[next++] = need[i];
	}
	if (eds2)
		ext[next++] = "VK_EXT_extended_dynamic_state2";
	free(exts);
	VkPhysicalDeviceTransformFeedbackFeaturesEXT xf = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT,
							   .transformFeedback = VK_TRUE};
	VkPhysicalDeviceExtendedDynamicState2FeaturesEXT e2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT,
							       .pNext = &xf, .extendedDynamicState2 = VK_TRUE};
	VkPhysicalDeviceDynamicRenderingFeaturesKHR dr = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR,
							  .pNext = eds2 ? (void *)&e2 : (void *)&xf, .dynamicRendering = VK_TRUE};
	float prio = 1.0f;
	VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1, .pQueuePriorities = &prio};
	VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &dr, .queueCreateInfoCount = 1,
				  .pQueueCreateInfos = &qci, .enabledExtensionCount = next, .ppEnabledExtensionNames = ext};
	CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
	vkGetDeviceQueue(dev, 0, 0, &queue);
	PROC(bind_xfb, "vkCmdBindTransformFeedbackBuffersEXT");
	PROC(begin_xfb, "vkCmdBeginTransformFeedbackEXT");
	PROC(end_xfb, "vkCmdEndTransformFeedbackEXT");
	PROC(draw_byte_count, "vkCmdDrawIndirectByteCountEXT");
	PROC(begin_query, "vkCmdBeginQueryIndexedEXT");
	PROC(end_query, "vkCmdEndQueryIndexedEXT");
	if (eds2)
		PROC(set_discard, "vkCmdSetRasterizerDiscardEnableEXT");
	PROC(begin_rendering, "vkCmdBeginRenderingKHR");
	PROC(end_rendering, "vkCmdEndRenderingKHR");
	if (!bind_xfb || !begin_xfb || !end_xfb || !begin_rendering || !end_rendering ||
	    (have_draw && !draw_byte_count) || (have_queries && (!begin_query || !end_query))) {
		fprintf(stderr, "xfb-test: missing entry points\n");
		return 3;
	}
	VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
				       .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
	CHECK(vkCreateCommandPool(dev, &pci, NULL, &pool));
	VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	CHECK(vkCreatePipelineLayout(dev, &li, NULL, &layout));
	struct target t = target();

	if (run3_mode)
		run3(&t);
	else if (ctl_mode || dup_spec)
		ladder(&t, ctl_mode, dup_spec);
	else {
		VkPipeline rec = pipeline(xfb_rec_vert, sizeof(xfb_rec_vert), false, true);
		VkPipeline inst_p = pipeline(xfb_inst_vert, sizeof(xfb_inst_vert), false, false);
		static const struct xcase cases[] = {
			{"rec", 6, 1, 0, 256, false},
			{"first3", 6, 1, 3, 256, false},
			{"overflow", 6, 1, 0, 48, true},
		};
		for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
			if (!only || !strcmp(only, cases[i].name))
				capture_case(&cases[i], rec, &t);
		if (!only || !strcmp(only, "instanced")) {
			static const struct xcase k = {"instanced", 3, 2, 0, 256, true};
			capture_case(&k, inst_p, &t);
		}
		if (only && !strcmp(only, "attr16"))
			attr16_case(&t);
		if (!only || !strcmp(only, "attr"))
			attr_case(&t);
		if (!only || !strcmp(only, "discard"))
			discard_case(&t);
		if (!only || !strcmp(only, "resume"))
			resume_case(rec, &t);
		if (!only || !strcmp(only, "dup"))
			failed |= dup_draw(&t, "dup", 3000, true, true, "case=");
		vkDestroyPipeline(dev, rec, NULL);
		vkDestroyPipeline(dev, inst_p, NULL);
		printf("result=%s\n", failed ? "FAIL" : offline ? "PASS-OFFLINE" : "PASS");
	}
	vkDestroyImageView(dev, t.view, NULL);
	vkDestroyImage(dev, t.img, NULL);
	vkFreeMemory(dev, t.m, NULL);
	vkDestroyPipelineLayout(dev, layout, NULL);
	vkDestroyCommandPool(dev, pool, NULL);
	vkDestroyDevice(dev, NULL);
	vkDestroyInstance(inst, NULL);
	return failed ? 1 : 0;
}
