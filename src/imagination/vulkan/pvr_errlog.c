/* SPDX-License-Identifier: MIT */
#include "pvr_errlog.h"
#include "pvr_device.h"
#include "pvr_debug.h"
#include "pvr_g5p_shader_dump.h"
#include "util/log.h"
#include "vk_alloc.h"
#include "vk_log.h"
#include "vk_command_buffer.h"
#include "vk_queue.h"
#include "vk_enum_to_str.h"
#include <fcntl.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

static void pvr_errlog_result(const char *entry, VkResult result)
{
   mesa_logw("pvr: %s returned %s", entry, vk_Result_to_str(result));
   const char *dir = pvr_g5p_shader_dump_dir();
   if (!dir)
      return;
   char path[4096], line[640];
   snprintf(path, sizeof(path), "%s/errors.tsv", dir);
   int n = snprintf(line, sizeof(line), "%lld\t%ld\tresult\t%s\t%s\n",
                    (long long)time(NULL), (long)getpid(), vk_Result_to_str(result), entry);
   int fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
   if (fd >= 0) {
      ssize_t written = write(fd, line, MIN2(n, (int)sizeof(line) - 1));
      (void)written;
      close(fd);
   }
}

#include "pvr_errlog_gen.h"

VkResult pvr_errlog_install(struct pvr_device *device)
{
   if (!PVR_IS_DEBUG_SET(ERRORS))
      return VK_SUCCESS;
   device->errlog_orig = vk_alloc(&device->vk.alloc, sizeof(*device->errlog_orig),
                                  8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!device->errlog_orig)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   pvr_errlog_patch(&device->vk.dispatch_table, device->errlog_orig);
   return VK_SUCCESS;
}
