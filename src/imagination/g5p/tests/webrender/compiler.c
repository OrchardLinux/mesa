/* SPDX-License-Identifier: MIT
 * Generated-by: LLM
 * Offline CLI for the merged driver's actual SPIR-V frontend and backend.
 */
#include "pvr_g5p.h"
#include "pco_g5p_general.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
   if (argc != 6) return 2; /* stage, spv, output, key, raw32 */
   FILE *f = fopen(argv[2], "rb"); if (!f) return 2;
   fseek(f, 0, SEEK_END); long bytes = ftell(f); rewind(f);
   if (bytes < 20 || bytes % 4) return 2;
   uint32_t *spv = malloc(bytes); if (!spv) return 2;
   if (fread(spv, 1, bytes, f) != (size_t)bytes) return 2;
   fclose(f);
   VkShaderModuleCreateInfo module = { .sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize=bytes, .pCode=spv };
   VkPipelineShaderStageCreateInfo stage = { .sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext=&module, .stage=!strcmp(argv[1],"vert") ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT, .pName="main" };
   struct pco_g5p_fs_key key = PCO_G5P_FS_KEY_DEFAULT;
   int k=atoi(argv[4]); key.last_read_groups = k != 0 && k != 4;
   if (k == 2 || k == 4 || k == 5) { key.blend=true; key.src_rgb=VK_BLEND_FACTOR_ONE; key.dst_rgb=VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      key.src_a=VK_BLEND_FACTOR_ONE; key.dst_a=VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; }
   if (k == 3) { key.blend=true; key.src_rgb=VK_BLEND_FACTOR_SRC_ALPHA; key.dst_rgb=VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      key.src_a=VK_BLEND_FACTOR_ONE; key.dst_a=VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; }
#ifdef DGL_RAW_OUTPUT
   key.color_output = atoi(argv[5]);
#endif
   key.fuse_ffma = k == 5 || k == 6;
   if (k >= 7) {
      key.blend = true;
      key.src_rgb = key.src_a = k == 7 ? VK_BLEND_FACTOR_ONE : VK_BLEND_FACTOR_ZERO;
      key.dst_rgb = key.dst_a = k == 7 ? VK_BLEND_FACTOR_ONE :
         k == 8 ? VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR : VK_BLEND_FACTOR_SRC_COLOR;
   }
   struct pco_g5p_program out = {0};
   bool ok = pvr_g5p_compile_stage(&stage, true, 1, NULL, &key, NULL, &out);
   free(spv);
   if (!ok) { fprintf(stderr,"REFUSED %s\n",out.refusal); return 77; }
   f=fopen(argv[3],"wb"); if (!f) return 2;
   bool wrote=fwrite(out.code,1,out.size,f)==out.size; fclose(f); free(out.code);
   return wrote ? 0 : 2;
}
