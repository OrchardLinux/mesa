/* SPDX-License-Identifier: MIT
 * Pure upload planning; no GPU/CPU mapping or descriptor contents are read.
 * The includer supplies pco_g5p_upload and its kind enum.
 */
#ifndef PVR_G5P_VTEX_PLAN_H
#define PVR_G5P_VTEX_PLAN_H
#include <stdbool.h>
#include <stddef.h>
#define VTEX_COMMON 256
#define VTEX_DMAS 32
struct vtex_dma {
   unsigned common, dwords;
   int upload; /* -1 CPU staging; otherwise original GPU UBO/XFB source */
};
struct pvr_g5p_vtex_uploads {
   unsigned count, dmas;
   struct pco_g5p_upload u[PCO_G5P_MAX_UPLOADS];
   struct vtex_dma dma[VTEX_DMAS];
};

static inline const char *
pvr_g5p_vtex_plan(const struct pco_g5p_upload *uploads, unsigned count, unsigned common,
                   struct pvr_g5p_vtex_uploads *plan)
{
   if (!count || count > PCO_G5P_MAX_UPLOADS || !common || common > VTEX_COMMON)
      return "invalid upload/common-store count";
   bool occupied[VTEX_COMMON] = {false}, cpu[VTEX_COMMON] = {false};
   plan->count = count;
   plan->dmas = 0;
   for (unsigned i = 0; i < count; i++) {
      const struct pco_g5p_upload *u = &uploads[i];
      if (!u->dwords || u->common >= common || u->dwords > common - u->common)
         return "invalid common-store upload interval";
      switch (u->kind) {
      case PCO_G5P_UPLOAD_IMAGE: case PCO_G5P_UPLOAD_SAMPLER: case PCO_G5P_UPLOAD_IMAGE_SIZE:
      case PCO_G5P_UPLOAD_PUSH: case PCO_G5P_UPLOAD_UBO: case PCO_G5P_UPLOAD_UBO_ADDR:
      case PCO_G5P_UPLOAD_SSBO_ADDR: case PCO_G5P_UPLOAD_XFB:
         break;
      default:
         return "unqualified source kind";
      }
      bool direct = u->kind == PCO_G5P_UPLOAD_UBO || u->kind == PCO_G5P_UPLOAD_XFB;
      for (unsigned k = u->common; k < u->common + u->dwords; k++) {
         if (occupied[k]) return "overlapping common-store uploads";
         occupied[k] = true;
         cpu[k] = !direct;
      }
      plan->u[i] = *u;
      if (direct) {
         if (plan->dmas == VTEX_DMAS) return "more than 32 direct PDS transfers";
         plan->dma[plan->dmas++] = (struct vtex_dma){u->common, u->dwords, i};
      }
   }
   for (unsigned k = 0; k < common;) {
      if (!cpu[k]) { k++; continue; }
      unsigned first = k++;
      while (k < common && cpu[k]) k++;
      if (plan->dmas == VTEX_DMAS) return "more than 32 coalesced PDS transfers";
      plan->dma[plan->dmas++] = (struct vtex_dma){first, k - first, -1};
   }
   return NULL;
}
#endif
