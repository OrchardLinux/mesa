/* SPDX-License-Identifier: MIT
 * Generated-by: LLM
 * Bounded fragment common-store layout, independent of NIR and USC encoders.
 */
#ifndef G5P_FS_COMMON_H
#define G5P_FS_COMMON_H
#include <stdbool.h>
#include <stdint.h>

struct g5p_fs_common_plan {
   bool memory_ubos, low_pointers, compact;
   unsigned pointer_end, end;
};

static inline unsigned g5p_common_align(unsigned n, unsigned a)
{
   return (n + a - 1) & ~(a - 1);
}

static inline bool
g5p_fs_common_plan(unsigned start, unsigned textures, unsigned dynamic_ptrs,
                   unsigned size_queries, unsigned ubos, uint64_t ubo_dwords,
                   uint64_t push_dwords, struct g5p_fs_common_plan *p)
{
   if (start > 4 || textures > 16 || dynamic_ptrs > 16 || size_queries > 4 || ubos > 8)
      return false;
   *p = (struct g5p_fs_common_plan){0};
   /* Each retry only enables one of two monotone choices. A layout which
    * already fits keeps its exact old descriptor/pointer positions. */
   for (unsigned retry = 0; retry < 4; retry++) {
      unsigned next = start;
      p->compact = textures > 4 || p->low_pointers;
      if (p->low_pointers) {
         next = g5p_common_align(next, 2) + 2 * (dynamic_ptrs + (p->memory_ubos ? ubos : 0));
         if (next > 32)
            return false;
         p->pointer_end = next;
      }
      if (textures && p->compact) {
         next = g5p_common_align(next, 8) + 10 * textures;
      } else {
         for (unsigned i = 0; i < textures; i++)
            next = g5p_common_align(next, 8) + 10;
      }
      unsigned dynamic_end = next;
      if (!p->low_pointers)
         dynamic_end = next = g5p_common_align(next, 2) + 2 * dynamic_ptrs;
      if (next < 2)
         next = 2;
      next += 4 * size_queries;
      if (!p->memory_ubos && ubos && (uint64_t)next + ubo_dwords + push_dwords > 256) {
         p->memory_ubos = true;
         continue;
      }
      if (!p->low_pointers && ((dynamic_ptrs && dynamic_end > 32) ||
          (p->memory_ubos && g5p_common_align(next, 2) + 2 * ubos > 32))) {
         p->low_pointers = true;
         continue;
      }
      if (!p->low_pointers && p->memory_ubos)
         next = g5p_common_align(next, 2) + 2 * ubos;
      uint64_t end = next + push_dwords + (p->memory_ubos ? 0 : ubo_dwords);
      if (end > 256)
         return false;
      p->end = end;
      return true;
   }
   return false;
}
#endif
