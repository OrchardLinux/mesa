/* SPDX-License-Identifier: MIT
 * Churn completed framebuffers while a different target's job is queued on the same context.
 * Valid Vulkan: the previous target is complete before it is destroyed; only the new target is pending.
 * Run with A9_FAKE_RT_BUSY=1 to defer the fake's context retirement until a successful fence wait.
 * The old winsys loses the refused destroy and reaches the kernel's 32-target cap; fixed winsys waits/retries.
 * No pixel claim: this checks target lifetime and submission, on fake or hardware. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>
#define CHECK(x) do { VkResult r=(x); if(r != VK_SUCCESS) { fprintf(stderr,"RT-CHURN FAIL %s = %d at %d\n",#x,r,__LINE__); exit(1); } } while(0)
struct target { VkImage image; VkDeviceMemory memory; VkImageView view; VkFramebuffer fb; };
static VkDevice dev;
static VkRenderPass rp;
static VkPhysicalDeviceMemoryProperties mp;
static struct target make_target(unsigned width)
{
   struct target t={0};
   VkImageCreateInfo ii={.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,.imageType=VK_IMAGE_TYPE_2D,
      .format=VK_FORMAT_B8G8R8A8_UNORM,.extent={width,64,1},.mipLevels=1,.arrayLayers=1,
      .samples=VK_SAMPLE_COUNT_1_BIT,.tiling=VK_IMAGE_TILING_LINEAR,.usage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
   CHECK(vkCreateImage(dev,&ii,0,&t.image));
   VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev,t.image,&mr);
   unsigned type=0; while(type<mp.memoryTypeCount && !(mr.memoryTypeBits & (1u<<type))) type++;
   if(type==mp.memoryTypeCount) exit(2);
   VkMemoryAllocateInfo ai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=mr.size,.memoryTypeIndex=type};
   CHECK(vkAllocateMemory(dev,&ai,0,&t.memory)); CHECK(vkBindImageMemory(dev,t.image,t.memory,0));
   VkImageViewCreateInfo vi={.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,.image=t.image,
      .viewType=VK_IMAGE_VIEW_TYPE_2D,.format=ii.format,.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};
   CHECK(vkCreateImageView(dev,&vi,0,&t.view));
   VkFramebufferCreateInfo fi={.sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,.renderPass=rp,
      .attachmentCount=1,.pAttachments=&t.view,.width=width,.height=64,.layers=1};
   CHECK(vkCreateFramebuffer(dev,&fi,0,&t.fb)); return t;
}
static void destroy_target(struct target t)
{
   if(!t.fb) return;
   vkDestroyFramebuffer(dev,t.fb,0); vkDestroyImageView(dev,t.view,0);
   vkDestroyImage(dev,t.image,0); vkFreeMemory(dev,t.memory,0);
}
int main(int argc,char **argv)
{
   unsigned count=argc>1?(unsigned)strtoul(argv[1],0,10):96;
   if(count<33) return 2;
   VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.apiVersion=VK_API_VERSION_1_0};
   VkInstanceCreateInfo ici={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};
   VkInstance inst; CHECK(vkCreateInstance(&ici,0,&inst));
   uint32_t n=1; VkPhysicalDevice pd;
   VkResult er=vkEnumeratePhysicalDevices(inst,&n,&pd);
   if((er!=VK_SUCCESS && er!=VK_INCOMPLETE)||!n) return 2;
   vkGetPhysicalDeviceMemoryProperties(pd,&mp);
   float priority=1;
   VkDeviceQueueCreateInfo qi={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueCount=1,.pQueuePriorities=&priority};
   VkDeviceCreateInfo di={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.queueCreateInfoCount=1,.pQueueCreateInfos=&qi};
   CHECK(vkCreateDevice(pd,&di,0,&dev));
   VkQueue queue; vkGetDeviceQueue(dev,0,0,&queue);
   VkAttachmentDescription ad={.format=VK_FORMAT_B8G8R8A8_UNORM,.samples=VK_SAMPLE_COUNT_1_BIT,
      .loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,.storeOp=VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,.finalLayout=VK_IMAGE_LAYOUT_GENERAL};
   VkAttachmentReference ref={0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
   VkSubpassDescription sub={.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,.colorAttachmentCount=1,.pColorAttachments=&ref};
   VkRenderPassCreateInfo ri={.sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,.attachmentCount=1,.pAttachments=&ad,.subpassCount=1,.pSubpasses=&sub};
   CHECK(vkCreateRenderPass(dev,&ri,0,&rp));
   VkCommandPoolCreateInfo pci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
   VkCommandPool pool; CHECK(vkCreateCommandPool(dev,&pci,0,&pool));
   VkFenceCreateInfo fci={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence; CHECK(vkCreateFence(dev,&fci,0,&fence));
   struct target previous={0};
   for(unsigned i=0;i<count;i++) {
      CHECK(vkResetCommandPool(dev,pool,0));
      struct target current=make_target(64+i);
      VkCommandBufferAllocateInfo ca={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=pool,
         .level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
      VkCommandBuffer cb; CHECK(vkAllocateCommandBuffers(dev,&ca,&cb));
      VkCommandBufferBeginInfo bi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
      CHECK(vkBeginCommandBuffer(cb,&bi));
      VkClearValue clear={.color={{0.2,0.4,0.6,1}}};
      VkRenderPassBeginInfo rb={.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,.renderPass=rp,.framebuffer=current.fb,
         .renderArea={{0,0},{64+i,64}},.clearValueCount=1,.pClearValues=&clear};
      vkCmdBeginRenderPass(cb,&rb,VK_SUBPASS_CONTENTS_INLINE); vkCmdEndRenderPass(cb);
      CHECK(vkEndCommandBuffer(cb)); CHECK(vkResetFences(dev,1,&fence));
      VkSubmitInfo si={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cb};
      CHECK(vkQueueSubmit(queue,1,&si,fence));
      /* Only current is pending. previous was waited on at the end of the preceding iteration. */
      destroy_target(previous);
      CHECK(vkWaitForFences(dev,1,&fence,VK_TRUE,5000000000ull));
      previous=current;
   }
   destroy_target(previous); vkDestroyFence(dev,fence,0); vkDestroyCommandPool(dev,pool,0);
   vkDestroyRenderPass(dev,rp,0); vkDestroyDevice(dev,0); vkDestroyInstance(inst,0);
   printf("RT-CHURN RESULT PASS %u targets submitted; completed targets destroyed beside pending jobs\n",count);
   return 0;
}
