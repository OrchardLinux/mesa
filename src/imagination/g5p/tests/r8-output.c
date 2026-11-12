/* SPDX-License-Identifier: MIT
 * Generated-by: LLM. Compile actual scalar R8 / RGBA8 DRAW/LOAD without a GPU pass.
 * r8-output {load|shader.spv} {r8ui|r8i|rgba8ui|rgba8i} output.usc
 * The host oracle compares terminal source counts with Apple's exports.
 */
#include "pvr_g5p.h"
#include "pco_g5p_general.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
   if (argc != 4)
      return 2;
   bool scalar = !strcmp(argv[2], "r8ui") || !strcmp(argv[2], "r8i");
   bool sign = !strcmp(argv[2], "r8i") || !strcmp(argv[2], "rgba8i");
   if (!scalar && strcmp(argv[2], "rgba8ui") && strcmp(argv[2], "rgba8i"))
      return 2;
   struct pco_g5p_program out = {0};
   bool ok;
   if (!strcmp(argv[1], "load")) {
      ok = scalar ? pvr_g5p_compile_load_scalar8(&out, sign)
                  : sign ? pvr_g5p_compile_load_packed_sint(&out, 8)
                         : pvr_g5p_compile_load_rgba8ui(&out);
   } else {
      FILE *f = fopen(argv[1], "rb");
      if (!f)
         return 2;
      fseek(f, 0, SEEK_END);
      long bytes = ftell(f);
      rewind(f);
      if (bytes < 20 || bytes % 4)
         return 2;
      uint32_t *spv = malloc(bytes);
      if (!spv || fread(spv, 1, bytes, f) != (size_t)bytes)
         return 2;
      fclose(f);
      VkShaderModuleCreateInfo module = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
         .codeSize = bytes, .pCode = spv };
      VkPipelineShaderStageCreateInfo stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .pNext = &module, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .pName = "main" };
      struct pco_g5p_fs_key key = PCO_G5P_FS_KEY_DEFAULT;
      key.last_read_groups = true;
      key.color_output = sign ? PCO_G5P_COLOR_SINT32 : PCO_G5P_COLOR_UINT32;
      key.packed_uint_bits = 8;
      key.packed_sint = sign;
      key.packed_scalar8 = scalar;
      ok = pvr_g5p_compile_stage(&stage, true, 1, NULL, &key, NULL, &out);
      free(spv);
   }
   if (!ok) {
      fprintf(stderr, "REFUSED %s\n", out.refusal);
      free(out.code);
      return 77;
   }
   printf("R8-OUTPUT format=%s bytes=%zu\n", argv[2], out.size);
   FILE *f = fopen(argv[3], "wb");
   if (!f)
      return 2;
   bool written = fwrite(out.code, 1, out.size, f) == out.size;
   fclose(f);
   free(out.code);
   return written ? 0 : 2;
}
