/*
 * Copyright © 2026 OrchardLinux
 * SPDX-License-Identifier: MIT
 *
 * Compiles shaders captured with G9P_CAPTURE (g9p_capture_failure()) again, off the
 * device: g9p_replay FILE... prints each one's result and code size.
 */
#include <stdio.h>
#include <stdlib.h>

#include "compiler/glsl_types.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_serialize.h"
#include "util/blob.h"
#include "util/ralloc.h"
#include "g9p_compile.h"

static bool
replay(const char *path)
{
   FILE *f = fopen(path, "rb");
   uint32_t head[2];
   struct g9p_capture_args args;
   uint64_t size;
   bool ok = false;

   if (!f || fread(head, sizeof(head), 1, f) != 1 || head[0] != G9P_CAPTURE_MAGIC ||
       head[1] != sizeof(args) || fread(&args, sizeof(args), 1, f) != 1 ||
       fread(&size, sizeof(size), 1, f) != 1) {
      fprintf(stderr, "%s: not a capture of this build\n", path);
      if (f)
         fclose(f);
      return false;
   }
   void *mem_ctx = ralloc_context(NULL);
   void *data = ralloc_size(mem_ctx, size);

   if (fread(data, size, 1, f) == 1) {
      struct blob_reader reader;
      struct g9p_shader out;

      blob_reader_init(&reader, data, size);
      nir_shader *nir = nir_deserialize(mem_ctx, g9p_nir_options(), &reader);

      switch (args.stage) {
      case MESA_SHADER_VERTEX:
         ok = g9p_compile_vertex(nir, args.attribs, args.frag_coord_slot, args.point_coord, &out,
                                 args.point_size, mem_ctx);
         break;
      case MESA_SHADER_FRAGMENT:
         ok = g9p_compile_fragment(nir, args.frag_coord_slot, args.has_blend ? &args.blend : NULL,
                                   args.has_coverage ? &args.coverage : NULL, args.targets, &out,
                                   mem_ctx);
         break;
      default:
         ok = g9p_compile_compute(nir, &out, mem_ctx);
         break;
      }
      printf("%s: %s", path, ok ? "ok" : "REFUSED");
      if (ok)
         printf(", %u bytes, %u registers", out.code_size, out.num_regs);
      printf("\n");
   }
   fclose(f);
   ralloc_free(mem_ctx);
   return ok;
}

int
main(int argc, char **argv)
{
   unsigned failed = 0;

   glsl_type_singleton_init_or_ref();
   for (int i = 1; i < argc; i++)
      failed += !replay(argv[i]);
   glsl_type_singleton_decref();
   return failed ? 1 : 0;
}
