/* SPDX-License-Identifier: MIT */
/*
 * Shader capture for compatibility triage (diagnostic; off unless set).
 *
 * PVR_DEBUG=shader_dump (PVR_SHADER_DUMP_DIR=<dir>): every shader stage the G5P backend compiles is
 * recorded in <dir>:
 *   <hash>.spv              the SPIR-V module, once (first 40 hex digits of
 *                           the BLAKE3 of its words)
 *   <hash>.<stage>.<v>.json the compile: stage, entry point, specialization
 *                           constants, pipeline layout, the backend's key
 *                           structs as hex (fragment key, vertex layout,
 *                           link; enough for an offline admission check
 *                           replay), and the result: accepted, or refused
 *                           with the backend's reason. <v> = a hash of
 *                           everything but the result.
 *   index.tsv               one line per compile: unix time, pid, hash,
 *                           stage, v, ADMIT/REFUSE/STUB, the reason
 * Files are written to a temporary name and renamed (several threads and
 * processes may compile at once); index lines are single appends.
 *
 * PVR_DEBUG=capture_stub: a refused stage is replaced by a stub with the
 * same interface (fragment: magenta at location 0; vertex: a point at the
 * origin, nothing drawn; compute: an empty kernel, the dispatch omitted), so
 * the pipeline is created and the application keeps creating the rest. The
 * refusal is still logged ("G5P capture stub: ...") and dumped. Never on by
 * default: stubbed pipelines draw wrong pictures.
 */
#ifndef PVR_G5P_SHADER_DUMP_H
#define PVR_G5P_SHADER_DUMP_H

#include <stdbool.h>
#include "c11/threads.h"
#include <stddef.h>
#include <vulkan/vulkan_core.h>

struct pvr_g5p_dump_blob {
   const char *name;
   const void *data;
   size_t size;
};

/* NULL when PVR_DEBUG=shader_dump is unset (read once) */
const char *pvr_g5p_shader_dump_dir(void);
bool pvr_g5p_capture_stub(void);
bool pvr_g5p_compute_log(void);
void pvr_g5p_shader_id(const VkPipelineShaderStageCreateInfo *info, char id[13]);

/* Record one compile of @stage_info (no-op without PVR_DEBUG=shader_dump).
 * @result: "ADMIT", "REFUSE" or "STUB" (refused, then stubbed). */
void pvr_g5p_shader_dump(const VkPipelineShaderStageCreateInfo *stage_info, VkPipelineLayout layout,
                         const char *stage, const struct pvr_g5p_dump_blob *blobs, unsigned nblobs,
                         const char *result, const char *reason);

/* A stage create info for the stub of @stage, in caller storage */
const VkPipelineShaderStageCreateInfo *pvr_g5p_stub_stage(VkShaderStageFlagBits stage,
                                                           VkPipelineShaderStageCreateInfo *info,
                                                           VkShaderModuleCreateInfo *module);

/* Pipelines the driver makes for itself (clear quads, vk_meta blits) are created with this
 * raised: a refused one fails as before instead of being admitted without programs */
extern thread_local unsigned pvr_g5p_internal_pipeline;

/* Set while the driver compiles a pipeline's one-layer variant: its stages' array samples are drawn as 2D
 * samples of a one-layer view (pvr_g5p_nir.c one_layer_as_2d()) */
extern thread_local bool pvr_g5p_one_layer_compile;

#endif
