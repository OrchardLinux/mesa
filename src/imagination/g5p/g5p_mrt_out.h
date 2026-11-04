/* SPDX-License-Identifier: MIT
 * N-target fragment output plan for the G5P (Apple A9) USC backend.
 *
 * Pure data, no NIR, Vulkan or UAPI types, so the same header is used by
 * pco_g5p_general.c and by the host tests (tools/mesa-a9-mrt107/).
 *
 * Evidence (docs/gpu/experiments/gles3-continued107.md section 0.3, Apple's
 * own A9 `metal-tt applegpu_g5p` output, never executed on a device):
 *   - Each 8-bit RGBA colour target takes two consecutive 32-bit output
 *     registers: o(2t) = {g:r}, o(2t+1) = {a:b}, as packed f16 halves.
 *   - One `emitpix` follows the output writes and carries the TOTAL word
 *     count, not a per-target count: 1 target `1c21 0008` (2), 2 targets
 *     `1c41 0008` (4), 4 targets `1c01 0008` (the 3-bit field wraps: field 0
 *     means 8). A mixed set (BGRA8+RGBA8+RGBA16F+R32F) gives `1c71 0008` (7).
 * Only the homogeneous 1/2/4 x 8-bit sets are planned here. Three targets
 * and mixed-format sets have no Apple witness and are refused by name.
 * Where the end-of-tile PBE/EOT side is concerned nothing in this file
 * decides anything: that is capture-gated (g5p_mrt_contract.h).
 */
#ifndef G5P_MRT_OUT_H
#define G5P_MRT_OUT_H

#include <stdbool.h>
#include <stdint.h>

#define G5P_MRT_MAX_TARGETS 4
#define G5P_MRT_FS_SLOTS 8 /* FRAG_RESULT_DATA0..DATA7 */
#define G5P_MRT_HALVES_PER_TARGET 4
#define G5P_MRT_MAX_HALVES (G5P_MRT_MAX_TARGETS * G5P_MRT_HALVES_PER_TARGET)
#define G5P_MRT_WORDS_PER_TARGET 2 /* 8-bit RGBA: two packed f16x2 dwords */
#define G5P_MRT_EMIT_FIELD_MASK 7u /* emitpix word-count field is 3 bits wide */

enum g5p_mrt_refusal {
   G5P_MRT_OK = 0,
   G5P_MRT_NO_TARGETS,
   G5P_MRT_TOO_MANY_TARGETS,
   G5P_MRT_THREE_TARGETS,
   G5P_MRT_UNWRITTEN_TARGET,
   G5P_MRT_BLEND,
   G5P_MRT_WRITE_MASK,
   G5P_MRT_MIXED_FORMATS,
};

static inline const char *g5p_mrt_refusal_str(enum g5p_mrt_refusal r)
{
   switch (r) {
   case G5P_MRT_OK:
      return "ok";
   case G5P_MRT_NO_TARGETS:
      return "no colour targets";
   case G5P_MRT_TOO_MANY_TARGETS:
      return "more than four colour targets";
   case G5P_MRT_THREE_TARGETS:
      return "three colour targets: Apple's compiler was only observed for 1, 2 and 4 "
             "(the pixel output width 6 is not a kernel width)";
   case G5P_MRT_UNWRITTEN_TARGET:
      return "a colour target has no output store while another has one (a per-target write mask "
             "is not characterized)";
   case G5P_MRT_BLEND:
      return "blending with several colour targets: the blend epilogue owns output halves 4..7 and 12, "
             "which are target 1's outputs";
   case G5P_MRT_WRITE_MASK:
      return "a colour write mask other than 0xf with several colour targets (read-modify-write merges "
             "are uncharacterized and the fragment key is per pipeline, not per target)";
   case G5P_MRT_MIXED_FORMATS:
      return "mixed colour formats: only homogeneous 8-bit RGBA sets are characterized (width 7 was "
             "observed for one mixed set, per-format word counts are not)";
   }
   return "?";
}

struct g5p_mrt_out_plan {
   unsigned targets;      /* 1, 2 or 4 */
   unsigned words;        /* emitpix word count: 2 * targets */
   unsigned emit_field;   /* the 3-bit instruction field: words & 7 (8 -> 0) */
   unsigned pixel_output_width; /* kernel/UAPI width (2, 4, 8) */
   unsigned first_half[G5P_MRT_MAX_TARGETS]; /* output half of target t component 0: 4t */
};

/* Output half of colour target `target`, component `comp` (0..3 = r,g,b,a).
 * Half 2d is the low half of output dword d. */
static inline unsigned g5p_mrt_half(unsigned target, unsigned comp)
{
   return target * G5P_MRT_HALVES_PER_TARGET + comp;
}

/* Output dword holding a half. */
static inline unsigned g5p_mrt_half_dword(unsigned half)
{
   return half / 2;
}

/* The emitpix word-count field. 8 wraps to 0 (Apple's `1c01`), which the
 * disassembler in tools/gpu-userspace/g5p_isa.py prints as words=0. */
static inline unsigned g5p_mrt_emit_field(unsigned words)
{
   return words & G5P_MRT_EMIT_FIELD_MASK;
}

static inline unsigned g5p_mrt_emit_words(unsigned field)
{
   field &= G5P_MRT_EMIT_FIELD_MASK;
   return field ? field : 8;
}

/* Word counts for which an Apple A9 witness exists for a homogeneous set. */
static inline bool g5p_mrt_emit_words_witnessed(unsigned words)
{
   return words == 2 || words == 4 || words == 8;
}

static inline enum g5p_mrt_refusal g5p_mrt_out_plan_make(unsigned targets, struct g5p_mrt_out_plan *p)
{
   *p = (struct g5p_mrt_out_plan){0};
   if (!targets)
      return G5P_MRT_NO_TARGETS;
   if (targets > G5P_MRT_MAX_TARGETS)
      return G5P_MRT_TOO_MANY_TARGETS;
   if (targets == 3)
      return G5P_MRT_THREE_TARGETS;
   p->targets = targets;
   p->words = targets * G5P_MRT_WORDS_PER_TARGET;
   p->emit_field = g5p_mrt_emit_field(p->words);
   p->pixel_output_width = p->words; /* [D] 2/4/8, matches the kernel table; mrt2/mrt4 capture confirms */
   for (unsigned t = 0; t < targets; t++)
      p->first_half[t] = g5p_mrt_half(t, 0);
   return G5P_MRT_OK;
}

/* A fragment location is an attachment's output iff its slot (location -
 * FRAG_RESULT_DATA0) is below the attachment count. Higher slots are
 * discarded: Zink broadcasts gl_FragColor to all eight locations, which is
 * correct for an N-attachment pass once the extra copies are dropped. A
 * location below N is never dropped and never moved. */
static inline bool g5p_mrt_slot_is_attached(unsigned slot, unsigned targets)
{
   return slot < targets;
}

/* `written[slot]` = component mask stored to FRAG_RESULT_DATA0 + slot after
 * unattached slots were dropped. Either no target is written (a colour
 * write-mask-0 program ends without emitpix) or every target is: one
 * emitpix writes all N targets for exactly the same pixels, so an
 * unwritten attachment would receive whatever the output registers hold. */
static inline enum g5p_mrt_refusal g5p_mrt_check_writes(const struct g5p_mrt_out_plan *p,
                                                       const uint8_t written[G5P_MRT_FS_SLOTS])
{
   unsigned with = 0;
   for (unsigned t = 0; t < p->targets; t++)
      with += (written[t] & 0xf) != 0;
   for (unsigned s = p->targets; s < G5P_MRT_FS_SLOTS; s++)
      if (written[s])
         return G5P_MRT_UNWRITTEN_TARGET; /* a stray unattached store must have been dropped first */
   if (with && with != p->targets)
      return G5P_MRT_UNWRITTEN_TARGET;
   return G5P_MRT_OK;
}

/* Programs with several targets take no blend and only the full write mask:
 * the premultiplied epilogue assumes one target (it uses destination halves
 * 4..7), and the one-attachment fragment key has no per-target state. A
 * zero mask would drop only target 0's stores and leave the others written. */
static inline enum g5p_mrt_refusal g5p_mrt_check_key(const struct g5p_mrt_out_plan *p, bool blend,
                                                    unsigned write_mask)
{
   if (p->targets == 1)
      return G5P_MRT_OK;
   if (blend)
      return G5P_MRT_BLEND;
   if (write_mask != 0xf)
      return G5P_MRT_WRITE_MASK;
   return G5P_MRT_OK;
}

#endif
