/* SPDX-License-Identifier: MIT */
/* AI-written: check the conservative limits-only branch on a real Vulkan device. */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL: %s\n",#x); return 1; } } while (0)
int main(void)
{
 VkInstance instance;
 VkInstanceCreateInfo ci = { .sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
 CHECK(vkCreateInstance(&ci, NULL, &instance)==VK_SUCCESS);
 uint32_t count=0;
 CHECK(vkEnumeratePhysicalDevices(instance,&count,NULL)==VK_SUCCESS && count);
 VkPhysicalDevice *devices=calloc(count,sizeof(*devices)); CHECK(devices);
 CHECK(vkEnumeratePhysicalDevices(instance,&count,devices)==VK_SUCCESS);
 unsigned found=0;
 for (unsigned i=0;i<count;i++) {
  VkPhysicalDeviceProperties p; VkPhysicalDeviceFeatures f;
  vkGetPhysicalDeviceProperties(devices[i],&p); vkGetPhysicalDeviceFeatures(devices[i],&f);
  if (!strstr(p.deviceName,"A10") && !strstr(p.deviceName,"G9P")) continue;
  found++;
  printf("device=%s targets=%u samples=0x%x\n",p.deviceName,p.limits.maxColorAttachments,p.limits.framebufferColorSampleCounts);
  CHECK(p.limits.maxColorAttachments==1 && p.limits.maxFragmentOutputAttachments==1);
  CHECK(p.limits.framebufferColorSampleCounts==VK_SAMPLE_COUNT_1_BIT);
  CHECK(p.limits.framebufferDepthSampleCounts==VK_SAMPLE_COUNT_1_BIT);
  CHECK(p.limits.framebufferStencilSampleCounts==VK_SAMPLE_COUNT_1_BIT);
  CHECK(p.limits.framebufferNoAttachmentsSampleCounts==VK_SAMPLE_COUNT_1_BIT);
  CHECK(p.limits.sampledImageColorSampleCounts==VK_SAMPLE_COUNT_1_BIT);
  CHECK(p.limits.sampledImageIntegerSampleCounts==VK_SAMPLE_COUNT_1_BIT);
  CHECK(p.limits.sampledImageDepthSampleCounts==VK_SAMPLE_COUNT_1_BIT);
  CHECK(p.limits.sampledImageStencilSampleCounts==VK_SAMPLE_COUNT_1_BIT);
  CHECK(!f.sampleRateShading && !f.alphaToOne);
  uint32_t n=0; CHECK(vkEnumerateDeviceExtensionProperties(devices[i],NULL,&n,NULL)==VK_SUCCESS);
  VkExtensionProperties *e=calloc(n?n:1,sizeof(*e)); CHECK(e);
  CHECK(vkEnumerateDeviceExtensionProperties(devices[i],NULL,&n,e)==VK_SUCCESS);
  for (unsigned j=0;j<n;j++) CHECK(strcmp(e[j].extensionName,VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME));
  free(e);
 }
 free(devices); vkDestroyInstance(instance,NULL);
 CHECK(found);
 puts("G9P-LIMITS RESULT PASS"); return 0;
}
