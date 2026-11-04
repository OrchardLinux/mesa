/* SPDX-License-Identifier: MIT */
/*
 * bind-budget: what the G5P winsys lets a process allocate and bind at once (zink10's nested Hyprland:
 * vkBindImageMemory failed while the arena and the 1 GiB window for own BOs still had room). Three parts:
 *  A  own BOs: 8 exportable (dma-buf) buffers of 16 MiB, each allocated and bound (128 MiB, above the arena's size);
 *  B  the arena: non-exportable buffers of 8 MiB until one fails (the winsys names the limit on stderr), then freed;
 *  C  the nested compositor's import: a 492x696 B8G8R8A8 LINEAR image exported from one instance (as a client's
 *     swapchain image) and imported into a second instance with an explicit LINEAR modifier and its row pitch, as
 *     Zink imports a client's dma-buf, then bound.
 * Prints one line per step and "BIND-BUDGET RESULT PASS|FAIL A=a/8 C=ok|<why>" (B is informational).
 *   ./bind-budget        (any Vulkan ICD; on the A9 over the fake node offline, or the device)
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#define MiB (1024ull * 1024)
#define DRM_FORMAT_MOD_LINEAR 0ull

struct ctx {
   VkInstance inst;
   VkPhysicalDevice pd;
   VkDevice dev;
   bool mods;
};

static int ctx_init(struct ctx *c)
{
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "bind-budget",
                            .apiVersion = VK_API_VERSION_1_1};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
   if (vkCreateInstance(&ici, NULL, &c->inst) != VK_SUCCESS)
      return -1;
   uint32_t n = 1;
   if (vkEnumeratePhysicalDevices(c->inst, &n, &c->pd) < 0 || !n)
      return -1;
   uint32_t ne = 0;
   vkEnumerateDeviceExtensionProperties(c->pd, NULL, &ne, NULL);
   VkExtensionProperties *e = calloc(ne, sizeof(*e));
   vkEnumerateDeviceExtensionProperties(c->pd, NULL, &ne, e);
   const char *want[] = {"VK_KHR_external_memory_fd", "VK_EXT_external_memory_dma_buf",
                         "VK_EXT_image_drm_format_modifier", "VK_KHR_image_format_list"};
   const char *on[4];
   unsigned non = 0;
   for (unsigned w = 0; w < 4; w++)
      for (uint32_t i = 0; i < ne; i++)
         if (!strcmp(e[i].extensionName, want[w])) {
            on[non++] = want[w];
            if (w == 2)
               c->mods = true;
         }
   free(e);
   float prio = 1;
   VkDeviceQueueCreateInfo q = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1,
                                .pQueuePriorities = &prio};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &q, .enabledExtensionCount = non, .ppEnabledExtensionNames = on};
   return vkCreateDevice(c->pd, &dci, NULL, &c->dev) == VK_SUCCESS ? 0 : -1;
}

static uint32_t mem_type(struct ctx *c, uint32_t bits)
{
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(c->pd, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if (bits & (1u << i))
         return i;
   return 0;
}

/* one buffer of size bytes, exportable or not: alloc and bind results */
static bool buffer(struct ctx *c, uint64_t size, bool exportable, VkBuffer *b, VkDeviceMemory *m, VkResult *ra,
                   VkResult *rb)
{
   VkExternalMemoryBufferCreateInfo ext = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                                           .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
   VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = exportable ? &ext : NULL,
                             .size = size, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
   *ra = *rb = VK_ERROR_UNKNOWN;
   if (vkCreateBuffer(c->dev, &bci, NULL, b) != VK_SUCCESS)
      return false;
   VkMemoryRequirements r;
   vkGetBufferMemoryRequirements(c->dev, *b, &r);
   VkExportMemoryAllocateInfo ex = {.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
                                    .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
   VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = exportable ? &ex : NULL,
                               .allocationSize = r.size, .memoryTypeIndex = mem_type(c, r.memoryTypeBits)};
   *ra = vkAllocateMemory(c->dev, &mai, NULL, m);
   if (*ra != VK_SUCCESS)
      return false;
   *rb = vkBindBufferMemory(c->dev, *b, *m, 0);
   return *rb == VK_SUCCESS;
}

/* the 492x696 image of part C: exporting (modifier list) or importing (explicit pitch, fd) */
static VkResult image(struct ctx *c, bool import, uint64_t pitch, int fd, VkImage *img, VkDeviceMemory *m,
                      VkResult *rb)
{
   const uint64_t mod = DRM_FORMAT_MOD_LINEAR;
   VkSubresourceLayout plane = {.offset = 0, .rowPitch = pitch};
   VkImageDrmFormatModifierListCreateInfoEXT list = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT, .drmFormatModifierCount = 1,
      .pDrmFormatModifiers = &mod};
   VkImageDrmFormatModifierExplicitCreateInfoEXT expl = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT, .drmFormatModifier = mod,
      .drmFormatModifierPlaneCount = 1, .pPlaneLayouts = &plane};
   VkExternalMemoryImageCreateInfo ext = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
                                          .pNext = import ? (void *)&expl : (void *)&list,
                                          .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
   VkImageCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .pNext = &ext,
                            .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_B8G8R8A8_UNORM,
                            .extent = {492, 696, 1}, .mipLevels = 1, .arrayLayers = 1,
                            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
                            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
   *rb = VK_ERROR_UNKNOWN;
   VkResult r = vkCreateImage(c->dev, &ici, NULL, img);
   if (r != VK_SUCCESS)
      return r;
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(c->dev, *img, &req);
   VkMemoryDedicatedAllocateInfo ded = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .image = *img};
   VkExportMemoryAllocateInfo ex = {.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, .pNext = &ded,
                                    .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
   VkImportMemoryFdInfoKHR im = {.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, .pNext = &ded,
                                 .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, .fd = fd};
   uint64_t size = req.size;
   if (import) {
      const off_t end = lseek(fd, 0, SEEK_END);
      lseek(fd, 0, SEEK_SET);
      if (end > 0 && (uint64_t)end > size)
         size = end;
   }
   VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                               .pNext = import ? (void *)&im : (void *)&ex, .allocationSize = size,
                               .memoryTypeIndex = mem_type(c, req.memoryTypeBits)};
   printf("C %s: image created, memory requirements %llu bytes, allocation %llu bytes\n",
          import ? "import" : "export", (unsigned long long)req.size, (unsigned long long)size);
   r = vkAllocateMemory(c->dev, &mai, NULL, m);
   if (r != VK_SUCCESS)
      return r;
   *rb = vkBindImageMemory(c->dev, *img, *m, 0);
   return VK_SUCCESS;
}

int main(void)
{
   setvbuf(stdout, NULL, _IOLBF, 0);
   struct ctx a = {0}, b = {0};
   if (ctx_init(&a)) {
      printf("BIND-BUDGET RESULT FAIL setup: no Vulkan device\n");
      return 3;
   }

   /* A: own BOs past the arena's size */
   VkBuffer ab[8];
   VkDeviceMemory am[8];
   unsigned bound = 0;
   for (unsigned i = 0; i < 8; i++) {
      VkResult ra, rb;
      bool ok = buffer(&a, 16 * MiB, true, &ab[i], &am[i], &ra, &rb);
      printf("A %u: exportable 16 MiB buffer alloc=%d bind=%d%s\n", i, ra, rb, ok ? "" : "  <- refused");
      if (ok)
         bound++;
   }
   printf("A: %u of 8 bound (%u MiB of own BOs)\n", bound, bound * 16);

   /* B: the arena's limit, informational */
   VkBuffer bb[16];
   VkDeviceMemory bm[16];
   unsigned nb = 0;
   for (; nb < 16; nb++) {
      VkResult ra, rb;
      if (!buffer(&a, 8 * MiB, false, &bb[nb], &bm[nb], &ra, &rb)) {
         printf("B: arena buffer %u (8 MiB) alloc=%d bind=%d: the limit after %u MiB\n", nb, ra, rb, nb * 8);
         if (ra == VK_SUCCESS)
            vkFreeMemory(a.dev, bm[nb], NULL);
         vkDestroyBuffer(a.dev, bb[nb], NULL);
         break;
      }
   }
   if (nb == 16)
      printf("B: 16 arena buffers of 8 MiB bound (128 MiB)\n");
   for (unsigned i = 0; i < nb; i++) {
      vkDestroyBuffer(a.dev, bb[i], NULL);
      vkFreeMemory(a.dev, bm[i], NULL);
   }

   /* C: export from instance a, import into instance b */
   const char *cwhy = "ok";
   if (!a.mods) {
      cwhy = "no VK_EXT_image_drm_format_modifier";
   } else if (ctx_init(&b)) {
      cwhy = "second instance";
   } else {
      VkImage ei, ii;
      VkDeviceMemory em, imem;
      VkResult rb;
      VkResult r = image(&a, false, 0, -1, &ei, &em, &rb);
      printf("C export: create/alloc=%d bind=%d\n", r, rb);
      if (r != VK_SUCCESS || rb != VK_SUCCESS) {
         cwhy = "export";
      } else {
         VkImageSubresource sub = {.aspectMask = VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT};
         VkSubresourceLayout lay;
         vkGetImageSubresourceLayout(a.dev, ei, &sub, &lay);
         PFN_vkGetMemoryFdKHR get_fd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(a.dev, "vkGetMemoryFdKHR");
         VkMemoryGetFdInfoKHR gi = {.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, .memory = em,
                                    .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
         int fd = -1;
         r = get_fd ? get_fd(a.dev, &gi, &fd) : VK_ERROR_EXTENSION_NOT_PRESENT;
         printf("C export: row pitch %llu, offset %llu, size %llu, fd=%d (%d)\n", (unsigned long long)lay.rowPitch,
                (unsigned long long)lay.offset, (unsigned long long)lay.size, fd, r);
         if (r != VK_SUCCESS) {
            cwhy = "vkGetMemoryFdKHR";
         } else {
            r = image(&b, true, lay.rowPitch, fd, &ii, &imem, &rb);
            printf("C import: create/alloc=%d bind=%d\n", r, rb);
            if (r != VK_SUCCESS)
               cwhy = "import create/alloc";
            else if (rb != VK_SUCCESS)
               cwhy = "import bind";
         }
      }
   }

   const bool pass = bound == 8 && !strcmp(cwhy, "ok");
   printf("BIND-BUDGET RESULT %s A=%u/8 C=%s\n", pass ? "PASS" : "FAIL", bound, cwhy);
   return pass ? 0 : 1;
}
