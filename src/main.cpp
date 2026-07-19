#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#define GLFW_INCLUDE_VULKAN        // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/hash.hpp>

// TINYGLTF_IMPLEMENTATION is already defined in the command line
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

// Include KTX library for texture loading
#include <ktx.h>

#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc_raii.hpp>

const uint32_t WIDTH  = 800;
const uint32_t HEIGHT = 600;
const std::string MODEL_PATH = "models/viking_room.glb";
const std::string TEXTURE_PATH = "textures/viking_room.ktx2";
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

struct DeviceCapabilities
{
  bool dynamicRenderingSupported = false;
  bool timelineSemaphoresSupported = false;
  bool synchronization2Supported = false;
};

struct Vertex
{
  glm::vec3 pos;
  glm::vec3 color;
  glm::vec2 texCoord;
  
  static vk::VertexInputBindingDescription getBindingDescription()
  {
    return {
      .binding = 0,
      .stride = sizeof(Vertex),
      .inputRate = vk::VertexInputRate::eVertex
    };
  }
  
  static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions()
  {
    return {{
      {.location = 0, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, pos)},
      {.location = 1, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, color)},
      {.location = 2, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(Vertex, texCoord)}
    }};
  }
  
  bool operator==(const Vertex &other) const {
    return pos == other.pos && color == other.color && texCoord == other.texCoord;
  }
};

template<> struct std::hash<Vertex>
{
  size_t operator() (Vertex const &vertex) const
  {
    return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
  }
};

struct UniformBufferObject
{
  alignas(16) glm::mat4 model;
  alignas(16) glm::mat4 view;
  alignas(16) glm::mat4 proj;
};

class Application
{
  
public:
  void run()
  {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }
  
private:
  GLFWwindow *window = nullptr;
  vk::raii::Context context;
  vk::raii::Instance instance = nullptr;
  vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
  vk::raii::SurfaceKHR surface = nullptr;
  vk::raii::PhysicalDevice physicalDevice = nullptr;
  DeviceCapabilities deviceCapabilities;
  vk::SampleCountFlagBits msaaSamples = vk::SampleCountFlagBits::e1;
  vk::raii::Device device = nullptr;
  uint32_t renderQueueIdx = ~0;
  vk::raii::Queue renderQueue = nullptr;
  vk::raii::SwapchainKHR swapchain = nullptr;
  std::vector<vk::Image> swapchainImages;
  vk::SurfaceFormatKHR swapchainSurfaceFormat;
  vk::Extent2D swapchainExtent;
  std::vector<vk::raii::ImageView> swapchainImageViews;
  
  vma::raii::Allocator allocator = nullptr;
  
  vk::raii::RenderPass renderPass = nullptr;
  vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
  vk::raii::PipelineLayout pipelineLayout = nullptr;
  vk::raii::Pipeline graphicsPipeline = nullptr;
  std::vector<vk::raii::Framebuffer> swapchainFramebuffers;
  
  vma::raii::Image colorImage = nullptr;
  vk::raii::ImageView colorImageView = nullptr;
  
  vma::raii::Image depthImage = nullptr;
  vk::raii::ImageView depthImageView = nullptr;
  
  uint32_t mipLevels = 0; // set by texture dimensions
  vma::raii::Image textureImage = nullptr;
  vk::raii::ImageView textureImageView = nullptr;
  vk::raii::Sampler textureSampler = nullptr;
  vk::Format textureImageFormat = vk::Format::eUndefined;
  
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  vma::raii::Buffer vertexBuffer = nullptr;
  vma::raii::Buffer indexBuffer = nullptr;
  
  std::vector<vma::raii::Buffer> uniformBuffers;
  std::vector<void *> uniformBuffersMapped;
  
  vk::raii::DescriptorPool descriptorPool = nullptr;
  std::vector<vk::raii::DescriptorSet> descriptorSets;
  
  vk::raii::CommandPool commandPool = nullptr;
  std::vector<vk::raii::CommandBuffer> commandBuffers;
  
  std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
  std::vector<vk::raii::Semaphore> renderCompleteSemaphores;
  std::vector<vk::raii::Fence> inFlightFences;
  
  vk::raii::Semaphore timelineSemaphore = nullptr;
  uint32_t frameIndex = 0;
  uint32_t frameResourceIndex = 0;
  
  bool framebufferResized = false;
  
  std::vector<const char *> requiredDeviceExtension = {
    vk::KHRSwapchainExtensionName,
    vk::KHRBufferDeviceAddressExtensionName,
    vk::EXTExtendedDynamicState2ExtensionName,
    vk::KHRShaderDrawParametersExtensionName,
#ifdef USE_VULKAN_PORTABILITY_FEATURES
    "VK_KHR_portability_subset"
#endif
  };
  
  void initWindow()
  {
    glfwInit();
    
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    
    window = glfwCreateWindow(WIDTH, HEIGHT, "VulkanSandbox", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
  }
  
  static void framebufferResizeCallback(GLFWwindow *window, int width, int height)
  {
    auto app = reinterpret_cast<Application *>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
  }
  
  void initVulkan()
  {
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    checkFeatureSupport();
    createLogicalDevice();
    createMemoryAllocator();
    createSwapchain();
    createImageViews();
    createColorResources();
    createDepthResources();
    if (!deviceCapabilities.dynamicRenderingSupported) {
      createRenderPass();
      createFramebuffers();
    }
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createCommandPool();
    createTextureImage();
    createTextureImageView();
    createTextureSampler();
    loadModel();
    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
    createSyncObjects();
    
    // Print feature support summary
    std::cout << "\nFeature support summary:\n";
    std::cout << "- Dynamic Rendering: " << (deviceCapabilities.dynamicRenderingSupported ? "Yes" : "No") << "\n";
    std::cout << "- Timeline Semaphores: " << (deviceCapabilities.timelineSemaphoresSupported ? "Yes" : "No") << "\n";
    std::cout << "- Synchronization2: " << (deviceCapabilities.synchronization2Supported ? "Yes" : "No") << "\n";
  }
  
  void mainLoop()
  {
    while (!glfwWindowShouldClose(window))
    {
      glfwPollEvents();
      drawFrame();
    }
    device.waitIdle(); // wait for device to finish operations before destroying resources
  }
  
  
  void cleanupSwapchain()
  {
    swapchainFramebuffers.clear();
    swapchainImageViews.clear();
    // semaphores tied to swapchain image indices need to be rebuilt on resize
    renderCompleteSemaphores.clear();
    swapchain = nullptr;
  }
  
  void cleanup()
  {
    glfwDestroyWindow(window);
    glfwTerminate();
  }
  
  void recreateSwapchain()
  {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
      glfwGetFramebufferSize(window, &width, &height);
      glfwWaitEvents();
    }
    
    device.waitIdle();
    
    cleanupSwapchain();
    createSwapchain();
    createImageViews();
    createColorResources();
    createDepthResources();
    if (!deviceCapabilities.dynamicRenderingSupported) {
      createRenderPass();
      createFramebuffers();
    }
    createGraphicsPipeline();
    // Recreate per-swapchain-image resources after resize
    createRenderCompleteSemaphores();
  }
  
  void createInstance()
  {
    const uint32_t apiVersion = std::min(context.enumerateInstanceVersion(), vk::ApiVersion14);
    const vk::ApplicationInfo appInfo{
      .pApplicationName = "Vulkan Sandbox",
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "No Engine",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = apiVersion
    };
    
    // Get required extensions
    auto extensions = getRequiredInstanceExtensions();
    
    vk::InstanceCreateFlags instanceFlags{};
#ifdef USE_VULKAN_PORTABILITY_FEATURES
    instanceFlags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
#endif

    // Create instance info
    vk::InstanceCreateInfo createInfo{
      .flags = instanceFlags,
      .pApplicationInfo = &appInfo,
      .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
      .ppEnabledExtensionNames = extensions.data()
    };

    instance = vk::raii::Instance(context, createInfo);
  }
  
  void setupDebugMessenger()
  {
    // Always set up debug messenger
    // Only used if validation layers enabled via vulkanconfig
    vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
                                                        vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                                                        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                                                        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
                                                        );
    vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
                                                       vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                                                       vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                                                       vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
                                                       );
    vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{
      .messageSeverity = severityFlags,
      .messageType = messageTypeFlags,
      .pfnUserCallback = debugCallback
    };
    try
    {
      debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
    }
    catch (vk::SystemError &err)
    {
      // If the debug utils extension is not available, this will fail
      // That's okay; it just means validation layers aren't enabled
      std::cout << "Debug messenger not available. Validation layers may not be enabled." << std::endl;
    }
  }
  
  void createSurface()
  {
    VkSurfaceKHR _surface; // C API Object
    if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
      throw std::runtime_error("failed to create window surface!");
    }
    surface = vk::raii::SurfaceKHR(instance, _surface); // promoted to C++ wrapper
  }
  
  
  void pickPhysicalDevice()
  {
    std::vector<vk::raii::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();
    
    // Use an ordered map to automatically sort candidates by increasing score
    std::multimap<int, vk::raii::PhysicalDevice> candidates;
    
    for (const auto& pd : physicalDevices) {
      auto deviceProperties = pd.getProperties();
      auto deviceFeatures = pd.getFeatures();
      uint32_t score = 0;
      
      // Discrete GPUs have a significant performance advantage
      if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
        score += 1000;
      }
      
      // Maximum possible size of textures affects graphics quality
      score += deviceProperties.limits.maxImageDimension2D;
      
      if (!isDeviceSuitable(pd)) continue;
      candidates.insert(std::make_pair(score, pd));
    }
    
    // Check if the best candidate is suitable at all
    if (!candidates.empty() && candidates.rbegin()->first > 0) {
      physicalDevice = candidates.rbegin()->second;
      msaaSamples = getMaxUsableSampleCount();
    } else {
      throw std::runtime_error("failed to find a suitable GPU!");
    }
  }

  void checkFeatureSupport()
  {
    // Get device properties to check Vulkan version
    vk::PhysicalDeviceProperties deviceProperties = physicalDevice.getProperties();

    // Get available extensions
    std::vector<vk::ExtensionProperties> availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();

    // Check for dynamic rendering support
    if (deviceProperties.apiVersion >= VK_VERSION_1_3)
    {
      deviceCapabilities.dynamicRenderingSupported = true;
      std::cout << "Dynamic rendering supported via Vulkan 1.3\n";
    }
    else
    {
      // Check for the extension on older Vulkan versions
      for (const auto &extension : availableExtensions)
      {
        if (strcmp(extension.extensionName, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME) == 0)
        {
          deviceCapabilities.dynamicRenderingSupported = true;
          std::cout << "Dynamic rendering supported via extension\n";
          break;
        }
      }
    }

    // Check for timeline semaphores support
    if (deviceProperties.apiVersion >= VK_VERSION_1_2)
    {
      deviceCapabilities.timelineSemaphoresSupported = true;
      std::cout << "Timeline semaphores supported via Vulkan 1.2\n";
    }
    else
    {
      // Check for the extension on older Vulkan versions
      for (const auto &extension : availableExtensions)
      {
        if (strcmp(extension.extensionName, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME) == 0)
        {
          deviceCapabilities.timelineSemaphoresSupported = true;
          std::cout << "Timeline semaphores supported via extension\n";
          break;
        }
      }
    }

    // Check for synchronization2 support
    if (deviceProperties.apiVersion >= VK_VERSION_1_3)
    {
      deviceCapabilities.synchronization2Supported = true;
      std::cout << "Synchronization2 supported via Vulkan 1.3\n";
    }
    else
    {
      // Check for the extension on older Vulkan versions
      for (const auto &extension : availableExtensions)
      {
        if (strcmp(extension.extensionName, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) == 0)
        {
          deviceCapabilities.synchronization2Supported = true;
          std::cout << "Synchronization2 supported via extension\n";
          break;
        }
      }
    }

    // Add required extensions based on feature support
    if (deviceCapabilities.dynamicRenderingSupported && deviceProperties.apiVersion < VK_VERSION_1_3)
    {
      requiredDeviceExtension.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
    }

    if (deviceCapabilities.timelineSemaphoresSupported && deviceProperties.apiVersion < VK_VERSION_1_2)
    {
      requiredDeviceExtension.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    }

    if (deviceCapabilities.synchronization2Supported && deviceProperties.apiVersion < VK_VERSION_1_3)
    {
      requiredDeviceExtension.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    }
  }
  
  void createLogicalDevice()
  {
    // find the index of the first queue family that supports graphics
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
    
    // get the first index into queueFamilyProperties which supports both graphics and presentation to window surface
    auto renderQueueFamilyIt = std::ranges::find_if(queueFamilyProperties,
                                                    [&](auto const &qfp) {
      uint32_t qfIdx = static_cast<uint32_t>(&qfp - queueFamilyProperties.data());
      bool supportsGraphics = (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0);
      bool supportsPresent = physicalDevice.getSurfaceSupportKHR(qfIdx, surface);
      return supportsGraphics && supportsPresent;
    });
    assert(renderQueueFamilyIt != queueFamilyProperties.end() && "Could not find queue family for graphics and presentation!");
    
    renderQueueIdx = static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), renderQueueFamilyIt));
    
    // create device with relevant features enabled
    vk::PhysicalDeviceFeatures2 enabledFeatures{
      .features = {
        .sampleRateShading = true,
        .samplerAnisotropy = true
      }
    };
    // Setup feature chain based on detected support
    void *pNext = nullptr;
    
    // Features required by application
    vk::PhysicalDeviceVulkan11Features vulkan11Features;
    vulkan11Features.shaderDrawParameters = true;
    vulkan11Features.pNext = pNext;
    pNext = &vulkan11Features;
    vk::PhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures;
    bufferDeviceAddressFeatures.bufferDeviceAddress = true;
    bufferDeviceAddressFeatures.pNext = pNext;
    pNext = &bufferDeviceAddressFeatures;
    vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT extendedDynamicStateFeatures;
    extendedDynamicStateFeatures.extendedDynamicState2 = true;
    extendedDynamicStateFeatures.pNext = pNext;
    pNext = &extendedDynamicStateFeatures;
    
    // Optional features
    vk::PhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures;
    vk::PhysicalDeviceSynchronization2Features synchronization2Features;
    vk::PhysicalDeviceTimelineSemaphoreFeatures timelineSemaphoreFeatures;
    if (deviceCapabilities.dynamicRenderingSupported) {
      dynamicRenderingFeatures.dynamicRendering = true;
      dynamicRenderingFeatures.pNext = pNext;
      pNext = &dynamicRenderingFeatures;
    }
    if (deviceCapabilities.synchronization2Supported) {
      synchronization2Features.synchronization2 = true;
      synchronization2Features.pNext = pNext;
      pNext = &synchronization2Features;
    }
    if (deviceCapabilities.timelineSemaphoresSupported) {
      timelineSemaphoreFeatures.timelineSemaphore = true;
      timelineSemaphoreFeatures.pNext = pNext;
      pNext = &timelineSemaphoreFeatures;
    }
    enabledFeatures.pNext = pNext;
    
    float queuePriority = 0.5f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo{
      .queueFamilyIndex = renderQueueIdx,
      .queueCount = 1,
      .pQueuePriorities = &queuePriority
    };
    vk::DeviceCreateInfo deviceCreateInfo{
      .pNext = &enabledFeatures,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &deviceQueueCreateInfo,
      .enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtension.size()),
      .ppEnabledExtensionNames = requiredDeviceExtension.data()
    };
    
    device = vk::raii::Device(physicalDevice, deviceCreateInfo);
    renderQueue = vk::raii::Queue(device, renderQueueIdx, 0);
  }
  
  void createMemoryAllocator()
  {
    vma::AllocatorCreateInfo allocatorInfo {
      .flags = vma::AllocatorCreateFlagBits::eBufferDeviceAddress,
      .physicalDevice = *physicalDevice,
      .vulkanApiVersion = physicalDevice.getProperties().apiVersion
    };
    allocator = vma::raii::Allocator(instance, device, allocatorInfo);
  }
  
  void createSwapchain()
  {
    vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
    swapchainExtent = chooseSwapExtent(surfaceCapabilities);
    uint32_t minImageCount = chooseSwapMinImageCount(surfaceCapabilities);
    
    std::vector<vk::SurfaceFormatKHR> surfaceFormats = physicalDevice.getSurfaceFormatsKHR(*surface);
    swapchainSurfaceFormat = chooseSwapSurfaceFormat(surfaceFormats);
    
    std::vector<vk::PresentModeKHR> availablePresentModes = physicalDevice.getSurfacePresentModesKHR(*surface);
    vk::PresentModeKHR presentMode = chooseSwapPresentMode(availablePresentModes);
    
    vk::SwapchainCreateInfoKHR swapchainCreateInfo{
      .surface = *surface,
      .minImageCount = minImageCount,
      .imageFormat = swapchainSurfaceFormat.format,
      .imageColorSpace = swapchainSurfaceFormat.colorSpace,
      .imageExtent = swapchainExtent,
      .imageArrayLayers = 1, // number of layers each image consists of
      .imageUsage = vk::ImageUsageFlagBits::eColorAttachment, // render directly to images (rather than intermediate framebuffers)
      .imageSharingMode = vk::SharingMode::eExclusive, // images owned by one queue family at a time
      .preTransform = surfaceCapabilities.currentTransform, // match existing display transformation
      .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque, // ignore alpha (for blending with other windows in windowing system)
      .presentMode = presentMode,
      .clipped = true, // ignore color of pixels obscured by another window
      .oldSwapchain = nullptr // reference to old swap chain needed for resizing, not needed now
    };
    
    swapchain       = vk::raii::SwapchainKHR(device, swapchainCreateInfo);
    swapchainImages = swapchain.getImages();
  }
  
  void createImageViews()
  {
    assert(swapchainImageViews.empty());
    swapchainImageViews.reserve(swapchainImages.size());
    
    for (const auto &image : swapchainImages) {
      swapchainImageViews.emplace_back(createImageView(image, swapchainSurfaceFormat.format, vk::ImageAspectFlagBits::eColor, 1));
    }
  }
  
  void createDescriptorSetLayout()
  {
    std::array<vk::DescriptorSetLayoutBinding, 2> bindings {{
      {
        .binding = 0,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eVertex
      },
      {
        .binding = 1,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment
      }
    }};
    vk::DescriptorSetLayoutCreateInfo layoutInfo {
      .bindingCount =  static_cast<uint32_t>(bindings.size()),
      .pBindings = bindings.data()
    };
    descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
  }
  
  void createGraphicsPipeline()
  {
    vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/shader.spv"));
    
    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
      .stage = vk::ShaderStageFlagBits::eVertex,
      .module = shaderModule,
      .pName = "vertMain",
      .pSpecializationInfo = nullptr // optionally specify constants for shader modules
    };
    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
      .stage = vk::ShaderStageFlagBits::eFragment,
      .module = shaderModule,
      .pName = "fragMain",
      .pSpecializationInfo = nullptr
    };
    vk::PipelineShaderStageCreateInfo shaderStagesInfo[] = {vertShaderStageInfo, fragShaderStageInfo};
    
    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo {
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &bindingDescription,
      .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
      .pVertexAttributeDescriptions = attributeDescriptions.data()
    };
    vk::PipelineInputAssemblyStateCreateInfo inputAssemblyInfo {.topology = vk::PrimitiveTopology::eTriangleList};
    vk::PipelineViewportStateCreateInfo viewportStateInfo {.viewportCount = 1, .scissorCount = 1};
    
    vk::PipelineRasterizationStateCreateInfo rasterizerInfo {
      .depthClampEnable = vk::False,
      .rasterizerDiscardEnable = vk::False,
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode = vk::CullModeFlagBits::eBack,
      .frontFace = vk::FrontFace::eCounterClockwise,
      .depthBiasEnable = vk::False,
      .lineWidth = 1.0f
    };
    
    // Multisampling combines fragments for the same pixel from multiple polygons to perform antialiasing cheaply
    vk::PipelineMultisampleStateCreateInfo multisamplingInfo {
      .rasterizationSamples = msaaSamples,
      .sampleShadingEnable = vk::True,
      .minSampleShading = 0.2f
    };
    
    vk::PipelineDepthStencilStateCreateInfo depthStencilInfo {
      .depthTestEnable = vk::True,
      .depthWriteEnable = vk::True,
      .depthCompareOp = vk::CompareOp::eLess,
      .depthBoundsTestEnable = vk::False,
      .stencilTestEnable = vk::False
    };
    
    // configure color blending (first struct per-framebuffer, second struct global)
    vk::PipelineColorBlendAttachmentState colorBlendAttachment {
      .blendEnable = vk::False,
      .colorWriteMask = vk::ColorComponentFlagBits::eR |
      vk::ColorComponentFlagBits::eG |
      vk::ColorComponentFlagBits::eB |
      vk::ColorComponentFlagBits::eA
    };
    vk::PipelineColorBlendStateCreateInfo colorBlendingInfo {
      .logicOpEnable = vk::False,
      .logicOp = vk::LogicOp::eCopy,
      .attachmentCount = 1,
      .pAttachments = &colorBlendAttachment
    };
    
    // enable dynamic viewport and scissor states (set in command buffer instead of during pipeline creation)
    std::vector<vk::DynamicState> dynamicStates = {
      vk::DynamicState::eViewport,
      vk::DynamicState::eScissor
    };
    
    vk::PipelineDynamicStateCreateInfo dynamicStateInfo {
      .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
      .pDynamicStates = dynamicStates.data()
    };
    
    // Create pipeline layout for uniform values (future use)
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo {
      .setLayoutCount = 1,
      .pSetLayouts = &*descriptorSetLayout,
      .pushConstantRangeCount = 0
    };
    pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);
    
    vk::GraphicsPipelineCreateInfo pipelineCreateInfo {
      .stageCount = 2,
      .pStages = shaderStagesInfo,
      .pVertexInputState = &vertexInputInfo,
      .pInputAssemblyState = &inputAssemblyInfo,
      .pViewportState = &viewportStateInfo,
      .pRasterizationState = &rasterizerInfo,
      .pMultisampleState = &multisamplingInfo,
      .pDepthStencilState = &depthStencilInfo,
      .pColorBlendState = &colorBlendingInfo,
      .pDynamicState = &dynamicStateInfo,
      .layout = pipelineLayout,
      .renderPass = deviceCapabilities.dynamicRenderingSupported ? nullptr : *renderPass
    };
    vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo {
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &swapchainSurfaceFormat.format,
      .depthAttachmentFormat = findDepthFormat()
    };
    if (deviceCapabilities.dynamicRenderingSupported) {
      std::cout << "Configuring graphics pipeline for dynamic rendering" << std::endl;
      pipelineCreateInfo.pNext = &pipelineRenderingCreateInfo;
    } else {
      std::cout << "Configuring graphics pipeline for traditional render pass" << std::endl;
    }

    graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfo);
  }

  void createRenderPass()
  {
    // Only called if dynamic rendering is not supported
    vk::AttachmentDescription colorAttachment {
      .format = swapchainSurfaceFormat.format,
      .samples = msaaSamples,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eStore,
      .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
      .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
      .initialLayout = vk::ImageLayout::eUndefined,
      .finalLayout = vk::ImageLayout::eColorAttachmentOptimal
    };
    vk::AttachmentDescription depthAttachment {
      .format = findDepthFormat(),
      .samples = msaaSamples,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eDontCare,
      .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
      .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
      .initialLayout = vk::ImageLayout::eUndefined,
      .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal
    };
    vk::AttachmentDescription colorAttachmentResolve {
      .format = swapchainSurfaceFormat.format,
      .samples = vk::SampleCountFlagBits::e1,
      .loadOp = vk::AttachmentLoadOp::eDontCare,
      .storeOp = vk::AttachmentStoreOp::eStore,
      .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
      .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
      .initialLayout = vk::ImageLayout::eUndefined,
      .finalLayout = vk::ImageLayout::ePresentSrcKHR
    };
    vk::AttachmentReference colorAttachmentRef {
      .attachment = 0,
      .layout = vk::ImageLayout::eColorAttachmentOptimal
    };
    vk::AttachmentReference depthAttachmentRef {
      .attachment = 1,
      .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal
    };
    vk::AttachmentReference colorAttachmentResolveRef {
      .attachment = 2,
      .layout = vk::ImageLayout::eColorAttachmentOptimal
    };
    vk::SubpassDescription subpass {
      .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
      .colorAttachmentCount = 1,
      .pColorAttachments = &colorAttachmentRef,
      .pResolveAttachments = &colorAttachmentResolveRef,
      .pDepthStencilAttachment = &depthAttachmentRef
    };
    vk::SubpassDependency dependency {
      .srcSubpass = VK_SUBPASS_EXTERNAL,
      .dstSubpass = 0,
      .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput |
                      vk::PipelineStageFlagBits::eEarlyFragmentTests,
      .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput |
                      vk::PipelineStageFlagBits::eEarlyFragmentTests,
      .srcAccessMask = vk::AccessFlagBits::eNone,
      .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite |
                       vk::AccessFlagBits::eDepthStencilAttachmentWrite
    };
    std::array<vk::AttachmentDescription, 3> attachments {
      colorAttachment,
      depthAttachment,
      colorAttachmentResolve
    };
    vk::RenderPassCreateInfo renderPassInfo {
      .attachmentCount = static_cast<uint32_t>(attachments.size()),
      .pAttachments = attachments.data(),
      .subpassCount = 1,
      .pSubpasses = &subpass,
      .dependencyCount = 1,
      .pDependencies = &dependency
    };
    renderPass = vk::raii::RenderPass(device, renderPassInfo);
  }

  void createFramebuffers()
  {
    // Only called if dynamic rendering is not supported
    swapchainFramebuffers.reserve(swapchainImageViews.size());
    for (auto const &swapchainImageView : swapchainImageViews) {
      std::array<vk::ImageView, 3> attachments {
        *colorImageView,
        *depthImageView,
        *swapchainImageView
      };
      vk::FramebufferCreateInfo framebufferInfo {
        .renderPass = *renderPass,
        .attachmentCount = static_cast<uint32_t>(attachments.size()),
        .pAttachments = attachments.data(),
        .width = swapchainExtent.width,
        .height = swapchainExtent.height,
        .layers = 1
      };
      swapchainFramebuffers.emplace_back(device, framebufferInfo);
    }
  }
  
  void createCommandPool()
  {
    vk::CommandPoolCreateInfo poolInfo {
      .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer, // allow command buffers to be rerecorded individually
      .queueFamilyIndex = renderQueueIdx
    };
    commandPool = vk::raii::CommandPool(device, poolInfo);
  }
  
  void createColorResources()
  {
    vk::Format colorFormat = swapchainSurfaceFormat.format;
    
    vk::ImageCreateInfo imageInfo {
      .imageType = vk::ImageType::e2D,
      .format = colorFormat,
      .extent = {swapchainExtent.width, swapchainExtent.height, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = msaaSamples,
      .tiling = vk::ImageTiling::eOptimal,
      .usage = vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
      .sharingMode = vk::SharingMode::eExclusive
    };
    vma::AllocationCreateInfo allocInfo {
      .flags = vma::AllocationCreateFlagBits::eDedicatedMemory,
      .usage = vma::MemoryUsage::eAutoPreferDevice
    };
    colorImage = vma::raii::Image(allocator, imageInfo, allocInfo);
    colorImageView = createImageView(colorImage, colorFormat, vk::ImageAspectFlagBits::eColor, 1);
  }

  
  void createDepthResources()
  {
    vk::Format depthFormat = findDepthFormat();

    vk::ImageCreateInfo imageInfo {
      .imageType = vk::ImageType::e2D,
      .format = depthFormat,
      .extent = {swapchainExtent.width, swapchainExtent.height, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = msaaSamples,
      .tiling = vk::ImageTiling::eOptimal,
      .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
      .sharingMode = vk::SharingMode::eExclusive
    };
    vma::AllocationCreateInfo allocInfo {
      .flags = vma::AllocationCreateFlagBits::eDedicatedMemory,
      .usage = vma::MemoryUsage::eAutoPreferDevice
    };
    depthImage = vma::raii::Image(allocator, imageInfo, allocInfo);
    depthImageView = createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);
  }

  vk::Format findSupportedFormat(const std::vector<vk::Format> &candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features)
  {
    for (const auto format : candidates)
    {
      vk::FormatProperties props = physicalDevice.getFormatProperties(format);
      if (((tiling == vk::ImageTiling::eLinear) && ((props.linearTilingFeatures & features) == features)) ||
          ((tiling == vk::ImageTiling::eOptimal) && ((props.optimalTilingFeatures & features) == features))) {
        return format;
      }
    }

    throw std::runtime_error("failed to find supported format!");
  }

  vk::Format findDepthFormat()
  {
    return findSupportedFormat({vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
                               vk::ImageTiling::eOptimal,
                               vk::FormatFeatureFlagBits::eDepthStencilAttachment);
  }
  
  void createTextureImage()
  {
    // Load KTX2 texture
    ktxTexture *kTexture;
    KTX_error_code result = ktxTexture_CreateFromNamedFile(TEXTURE_PATH.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &kTexture);
    if (result != KTX_SUCCESS) {
      throw std::runtime_error("Failed to load KTX texture image!");
    }
    
    // Get texture dimensions and data
    uint32_t texWidth = kTexture->baseWidth;
    uint32_t texHeight = kTexture->baseHeight;
    ktx_size_t imageSize = ktxTexture_GetImageSize(kTexture, 0);
    ktx_uint8_t *ktxTextureData = ktxTexture_GetData(kTexture);
    
    mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;
    vk::BufferCreateInfo stagingBufferInfo {
      .size = imageSize,
      .usage = vk::BufferUsageFlagBits::eTransferSrc,
      .sharingMode = vk::SharingMode::eExclusive
    };
    vma::AllocationCreateInfo stagingAllocInfo {
      .flags = vma::AllocationCreateFlagBits::eHostAccessSequentialWrite,
      .usage = vma::MemoryUsage::eAuto
    };
    vma::raii::Buffer stagingBuffer(allocator, stagingBufferInfo, stagingAllocInfo);
    stagingBuffer.getAllocation().copyFromMemory(ktxTextureData, 0, imageSize);
    
    vk::Format textureFormat;
    
    // Check if KTX texture has a format
    if (kTexture->classId == ktxTexture2_c) {
      // For KTX2 files, we can get the format directly
      auto *ktx2 = reinterpret_cast<ktxTexture2 *>(kTexture);
      textureFormat = static_cast<vk::Format>(ktx2->vkFormat);
      if (textureFormat == vk::Format::eUndefined) {
        // If the format is undefined, fall back to a reasonable default
        textureFormat = vk::Format::eR8G8B8A8Unorm;
      }
    } else {
      // For KTX1 files or if we can't determine the format, use a reasonable default
      textureFormat = vk::Format::eR8G8B8A8Unorm;
    }
    
    textureImageFormat = textureFormat;
    
    // create image in device memory as copy target
    vk::ImageCreateInfo imageInfo {
      .imageType = vk::ImageType::e2D,
      .format = textureImageFormat,
      .extent = {static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1},
      .mipLevels = mipLevels,
      .arrayLayers = 1,
      .samples = vk::SampleCountFlagBits::e1,
      .tiling = vk::ImageTiling::eOptimal,
      .usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
      .sharingMode = vk::SharingMode::eExclusive
    };
    vma::AllocationCreateInfo allocInfo {
      .usage = vma::MemoryUsage::eAutoPreferDevice
    };
    textureImage = vma::raii::Image(allocator, imageInfo, allocInfo);
    
    // copy pixel data from staging buffer to texture image and generate mipmaps
    vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();
    transitionImageLayout(
        commandBuffer,
        *textureImage,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eTransferDstOptimal,
        {},
        vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eTopOfPipe,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::ImageAspectFlagBits::eColor,
        0,
        mipLevels
    );
    copyBufferToImage(commandBuffer, stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
    generateMipmaps(commandBuffer, textureImage, vk::Format::eR8G8B8A8Srgb, texWidth, texHeight, mipLevels);
    endSingleTimeCommands(std::move(commandBuffer));
    
    ktxTexture_Destroy(kTexture);
  }
  
  void generateMipmaps(vk::raii::CommandBuffer &commandBuffer, vk::raii::Image &image, vk::Format imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels)
  {
    // check if image format supports linear blit-ing
    vk::FormatProperties formatProperties = physicalDevice.getFormatProperties(imageFormat);
    
    if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
      throw std::runtime_error("texture image format does not support linear blit-ting!");
    }
    
    int32_t mipWidth = texWidth;
    int32_t mipHeight = texHeight;
    
    for (uint32_t i = 1; i < mipLevels; i++) {
      transitionImageLayout(commandBuffer,
                            *image,
                            vk::ImageLayout::eTransferDstOptimal,
                            vk::ImageLayout::eTransferSrcOptimal,
                            vk::AccessFlagBits2::eTransferWrite,
                            vk::AccessFlagBits2::eTransferRead,
                            vk::PipelineStageFlagBits2::eTransfer,
                            vk::PipelineStageFlagBits2::eTransfer,
                            vk::ImageAspectFlagBits::eColor,
                            i - 1,
                            1);
      vk::ArrayWrapper1D<vk::Offset3D, 2> offsets, dstOffsets;
      offsets[0] = vk::Offset3D(0, 0, 0);
      offsets[1] = vk::Offset3D(mipWidth, mipHeight, 1);
      dstOffsets[0] = vk::Offset3D(0, 0, 0);
      dstOffsets[1] = vk::Offset3D(mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1);
      vk::ImageBlit blit = {
        .srcSubresource = {
          .aspectMask = vk::ImageAspectFlagBits::eColor,
          .mipLevel = i-1,
          .baseArrayLayer = 0,
          .layerCount = 1
        },
        .srcOffsets = offsets,
        .dstSubresource = {
          .aspectMask = vk::ImageAspectFlagBits::eColor,
          .mipLevel = i,
          .baseArrayLayer = 0,
          .layerCount = 1
        },
        .dstOffsets = dstOffsets
      };
      commandBuffer.blitImage(*image, vk::ImageLayout::eTransferSrcOptimal, *image, vk::ImageLayout::eTransferDstOptimal, {blit}, vk::Filter::eLinear);
      transitionImageLayout(commandBuffer,
                            *image,
                            vk::ImageLayout::eTransferSrcOptimal,
                            vk::ImageLayout::eShaderReadOnlyOptimal,
                            vk::AccessFlagBits2::eTransferRead,
                            vk::AccessFlagBits2::eShaderRead,
                            vk::PipelineStageFlagBits2::eTransfer,
                            vk::PipelineStageFlagBits2::eFragmentShader,
                            vk::ImageAspectFlagBits::eColor,
                            i - 1,
                            1);
      
      if (mipWidth > 1) mipWidth /= 2;
      if (mipHeight > 1) mipHeight /= 2;
    }
    transitionImageLayout(commandBuffer,
                          *image,
                          vk::ImageLayout::eTransferDstOptimal,
                          vk::ImageLayout::eShaderReadOnlyOptimal,
                          vk::AccessFlagBits2::eTransferWrite,
                          vk::AccessFlagBits2::eShaderRead,
                          vk::PipelineStageFlagBits2::eTransfer,
                          vk::PipelineStageFlagBits2::eFragmentShader,
                          vk::ImageAspectFlagBits::eColor,
                          mipLevels - 1,
                          1);
  }
  
  vk::SampleCountFlagBits getMaxUsableSampleCount()
  {
    vk::PhysicalDeviceProperties physicalDeviceProperties = physicalDevice.getProperties();

    vk::SampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
    if (counts & vk::SampleCountFlagBits::e64) { return vk::SampleCountFlagBits::e64; }
    if (counts & vk::SampleCountFlagBits::e32) { return vk::SampleCountFlagBits::e32; }
    if (counts & vk::SampleCountFlagBits::e16) { return vk::SampleCountFlagBits::e16; }
    if (counts & vk::SampleCountFlagBits::e8) { return vk::SampleCountFlagBits::e8; }
    if (counts & vk::SampleCountFlagBits::e4) { return vk::SampleCountFlagBits::e4; }
    if (counts & vk::SampleCountFlagBits::e2) { return vk::SampleCountFlagBits::e2; }

    return vk::SampleCountFlagBits::e1;
  }
  
  void createTextureImageView()
  {
    textureImageView = createImageView(*textureImage, textureImageFormat, vk::ImageAspectFlagBits::eColor, mipLevels);
  }
  
  void createTextureSampler()
  {
    vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
    vk::SamplerCreateInfo samplerInfo {
      .magFilter = vk::Filter::eLinear,
      .minFilter = vk::Filter::eLinear,
      .mipmapMode = vk::SamplerMipmapMode::eLinear,
      .addressModeU = vk::SamplerAddressMode::eRepeat,
      .addressModeV = vk::SamplerAddressMode::eRepeat,
      .addressModeW = vk::SamplerAddressMode::eRepeat,
      .mipLodBias = 0.0f,
      .anisotropyEnable = vk::True,
      .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
      .compareEnable = vk::False,
      .compareOp = vk::CompareOp::eAlways,
      .minLod = 0.0f,
      .maxLod = vk::LodClampNone
    };
    textureSampler = vk::raii::Sampler(device, samplerInfo);
  }
  
  vk::raii::ImageView createImageView(vk::Image const &image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels) const
  {
    vk::ImageViewCreateInfo viewInfo {
      .image = image,
      .viewType = vk::ImageViewType::e2D,
      .format = format,
      .subresourceRange = {
        .aspectMask = aspectFlags,
        .baseMipLevel = 0,
        .levelCount = mipLevels,
        .baseArrayLayer = 0,
        .layerCount = 1
      }
    };
    return vk::raii::ImageView(device, viewInfo);
  }
  
  void copyBufferToImage(vk::raii::CommandBuffer &commandBuffer, const vk::raii::Buffer &buffer, vk::raii::Image &image, uint32_t width, uint32_t height)
  {
    vk::BufferImageCopy region {
      .bufferOffset      = 0,
      .bufferRowLength   = 0,
      .bufferImageHeight = 0,
      .imageSubresource  = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = 1
      },
      .imageOffset       = {0, 0, 0},
      .imageExtent       = {width, height, 1}
    };
    commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
  }
  
  void loadModel()
  {
    // Use tinygltf to load the model
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;
    bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, MODEL_PATH);
    if (!warn.empty()) {
      std::cout << "glTF warning: " << warn << std::endl;
    }
    if (!err.empty()) {
        std::cout << "glTF error: " << err << std::endl;
    }
    if (!ret) {
        throw std::runtime_error("Failed to load glTF model");
    }
    
    const int sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;
    if (sceneIndex >= static_cast<int>(model.scenes.size())) {
      throw std::runtime_error("glTF model has no valid scene");
    }
    
    vertices.clear();
    indices.clear();
    
    for (int rootNodeIndex : model.scenes[sceneIndex].nodes) {
      loadNode(model, rootNodeIndex, glm::mat4{1.0f});
    }
    
    std::cout << "Loaded model with " << vertices.size() << " unique vertices and " << indices.size() << " indices." << std::endl;
  }
  
  void loadNode(tinygltf::Model const &model, int nodeIndex, glm::mat4 const &parentTransform)
  {
    auto const &node = model.nodes[nodeIndex];
    glm::mat4 const worldTransform =
      parentTransform * getNodeLocalTransform(node);

    if (node.mesh >= 0) {
      auto const &mesh = model.meshes[node.mesh];

      for (auto const &primitive : mesh.primitives) {
        loadPrimitive(model, primitive, worldTransform);
      }
    }

    for (int childIndex : node.children) {
      loadNode(model, childIndex, worldTransform);
    }
  }
  
  void loadPrimitive(tinygltf::Model const &model, tinygltf::Primitive const &primitive, glm::mat4 const &worldTransform)
  {
    if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
      return;
    }
    
    // Get indices
    const tinygltf::Accessor &indexAccessor = model.accessors[primitive.indices];
    const tinygltf::BufferView &indexBufferView = model.bufferViews[indexAccessor.bufferView];
    const tinygltf::Buffer &indexBuffer = model.buffers[indexBufferView.buffer];
    
    // Get vertex positions
    const tinygltf::Accessor &posAccessor = model.accessors[primitive.attributes.at("POSITION")];
    const tinygltf::BufferView &posBufferView = model.bufferViews[posAccessor.bufferView];
    const tinygltf::Buffer &posBuffer = model.buffers[posBufferView.buffer];
    
    
    // Get texture coordinates if available
    const bool hasTexCoords = primitive.attributes.contains("TEXCOORD_0");
    const tinygltf::Accessor *texCoordAccessor = nullptr;
    const tinygltf::BufferView *texCoordBufferView = nullptr;
    const tinygltf::Buffer *texCoordBuffer = nullptr;

    if (hasTexCoords) {
      texCoordAccessor = &model.accessors[primitive.attributes.at("TEXCOORD_0")];
      texCoordBufferView = &model.bufferViews[texCoordAccessor->bufferView];
      texCoordBuffer = &model.buffers[texCoordBufferView->buffer];
    }
    
    uint32_t const baseVertex = static_cast<uint32_t>(vertices.size());

    for (size_t i = 0; i < posAccessor.count; ++i) {
      const float *pos = reinterpret_cast<const float *>(&posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset + i * posAccessor.ByteStride(posBufferView)]);
      
      Vertex vertex{};
      // Vulkan's Y-axis correction already handled in UBO projection matrix inversion
      vertex.pos = glm::vec3(worldTransform * glm::vec4(pos[0], pos[1], pos[2], 1.0f));
      vertex.color = {1.0f, 1.0f, 1.0f};

      if (hasTexCoords) {
        const float *texCoord = reinterpret_cast<const float *>(&texCoordBuffer->data[texCoordBufferView->byteOffset + texCoordAccessor->byteOffset + i * texCoordAccessor->ByteStride(*texCoordBufferView)]);
        vertex.texCoord = {texCoord[0], texCoord[1]};
      } else {
        vertex.texCoord = {0.0f, 0.0f};
      }

      vertices.push_back(vertex);
    }
    
    const unsigned char *indexData = &indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset];
    size_t indexCount = indexAccessor.count;
    size_t indexStride = 0;
    
    // Determine index stride based on component type
    if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
      indexStride = sizeof(uint16_t);
    } else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
      indexStride = sizeof(uint32_t);
    } else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
      indexStride = sizeof(uint8_t);
    } else {
      throw std::runtime_error("Unsupported index component type");
    }
    
    indices.reserve(indices.size() + indexCount);
    
    for (size_t i = 0; i < indexCount; i++) {
      uint32_t index = 0;
      
      if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
        index = *reinterpret_cast<const uint16_t *>(indexData + i * indexStride);
      } else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
        index = *reinterpret_cast<const uint32_t *>(indexData + i * indexStride);
      } else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
        index = *reinterpret_cast<const uint8_t *>(indexData + i * indexStride);
      }
      
      indices.push_back(baseVertex + index);
    }
  }
  
  glm::mat4 getNodeLocalTransform(tinygltf::Node const &node) const
  {
    if (node.matrix.size() == 16) {
      glm::mat4 matrix{1.0f};

      // glTF matrices are stored column-major, as GLM expects.
      for (uint32_t column = 0; column < 4; ++column) {
        for (uint32_t row = 0; row < 4; ++row) {
          matrix[column][row] = static_cast<float>(node.matrix[column * 4 + row]);
        }
      }

      return matrix;
    }

    glm::mat4 transform{1.0f};

    if (node.translation.size() == 3) {
      transform = glm::translate(transform, glm::vec3(
        node.translation[0],
        node.translation[1],
        node.translation[2]));
    }

    if (node.rotation.size() == 4) {
      // glTF: x, y, z, w. GLM constructor: w, x, y, z.
      glm::quat rotation(
        node.rotation[3],
        node.rotation[0],
        node.rotation[1],
        node.rotation[2]);

      transform *= glm::mat4_cast(rotation);
    }

    if (node.scale.size() == 3) {
      transform = glm::scale(transform, glm::vec3(
        node.scale[0],
        node.scale[1],
        node.scale[2]));
    }

    return transform;
  }
  
  void createVertexBuffer()
  {
    vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
    
    vk::BufferCreateInfo stagingBufferInfo {
      .size = bufferSize,
      .usage = vk::BufferUsageFlagBits::eTransferSrc,
      .sharingMode = vk::SharingMode::eExclusive
    };
    vma::AllocationCreateInfo stagingAllocInfo {
      .flags = vma::AllocationCreateFlagBits::eHostAccessSequentialWrite,
      .usage = vma::MemoryUsage::eAuto,
    };
    vma::raii::Buffer stagingBuffer(allocator, stagingBufferInfo, stagingAllocInfo);
    stagingBuffer.getAllocation().copyFromMemory(vertices.data(), 0, bufferSize);
    
    vk::BufferCreateInfo vertexBufferInfo {
      .size = bufferSize,
      .usage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
      .sharingMode = vk::SharingMode::eExclusive
    };
    vma::AllocationCreateInfo vertexAllocInfo {
      .usage = vma::MemoryUsage::eAutoPreferDevice,
    };
    vertexBuffer = vma::raii::Buffer(allocator, vertexBufferInfo, vertexAllocInfo);
    copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
  }
  
  void createIndexBuffer()
  {
    vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();
    
    vk::BufferCreateInfo stagingBufferInfo {
      .size = bufferSize,
      .usage = vk::BufferUsageFlagBits::eTransferSrc,
      .sharingMode = vk::SharingMode::eExclusive
    };
    vma::AllocationCreateInfo stagingAllocInfo {
      .flags = vma::AllocationCreateFlagBits::eHostAccessSequentialWrite,
      .usage = vma::MemoryUsage::eAuto,
    };
    vma::raii::Buffer stagingBuffer(allocator, stagingBufferInfo, stagingAllocInfo);
    stagingBuffer.getAllocation().copyFromMemory(indices.data(), 0, bufferSize);
    vk::BufferCreateInfo indexBufferInfo {
      .size = bufferSize,
      .usage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
      .sharingMode = vk::SharingMode::eExclusive
    };
    vma::AllocationCreateInfo indexAllocInfo {
      .usage = vma::MemoryUsage::eAutoPreferDevice,
    };
    indexBuffer = vma::raii::Buffer(allocator, indexBufferInfo, indexAllocInfo);
    copyBuffer(stagingBuffer, indexBuffer, bufferSize);
  }
  
  void createUniformBuffers()
  {
    vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vk::BufferCreateInfo uniformBufferInfo {
        .size = bufferSize,
        .usage = vk::BufferUsageFlagBits::eUniformBuffer,
        .sharingMode = vk::SharingMode::eExclusive
      };
      vma::AllocationCreateInfo uniformAllocInfo {
        .flags = vma::AllocationCreateFlagBits::eHostAccessSequentialWrite | vma::AllocationCreateFlagBits::eMapped,
        .usage = vma::MemoryUsage::eAuto
      };
      vma::AllocationInfo uniformMemoryInfo{};
      uniformBuffers.emplace_back(allocator, uniformBufferInfo, uniformAllocInfo, &uniformMemoryInfo);
      uniformBuffersMapped.emplace_back(uniformMemoryInfo.pMappedData);
    }
  }
  
  void createDescriptorPool()
  {
    std::array<vk::DescriptorPoolSize, 2> poolSize {{
      {
        .type = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = MAX_FRAMES_IN_FLIGHT
      },
      {
        .type = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = MAX_FRAMES_IN_FLIGHT
      }
    }};
    vk::DescriptorPoolCreateInfo poolInfo {
      .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      .maxSets = MAX_FRAMES_IN_FLIGHT,
      .poolSizeCount = static_cast<uint32_t>(poolSize.size()),
      .pPoolSizes = poolSize.data()
    };
    descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
  }
  
  void createDescriptorSets()
  {
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo {
      .descriptorPool = descriptorPool,
      .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
      .pSetLayouts = layouts.data()
    };
    
    descriptorSets.clear();
    descriptorSets = device.allocateDescriptorSets(allocInfo);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vk::DescriptorBufferInfo bufferInfo {
        .buffer = uniformBuffers[i],
        .offset = 0,
        .range = sizeof(UniformBufferObject)
      };
      vk::DescriptorImageInfo imageInfo {
        .sampler = textureSampler,
        .imageView = textureImageView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
      };
      std::array<vk::WriteDescriptorSet, 2> descriptorWrites {{
        {
          .dstSet = descriptorSets[i],
          .dstBinding = 0,
          .dstArrayElement = 0,
          .descriptorCount = 1,
          .descriptorType = vk::DescriptorType::eUniformBuffer,
          .pBufferInfo = &bufferInfo
        },
        {
          .dstSet = descriptorSets[i],
          .dstBinding = 1,
          .dstArrayElement = 0,
          .descriptorCount = 1,
          .descriptorType = vk::DescriptorType::eCombinedImageSampler,
          .pImageInfo = &imageInfo
        }
      }};
      device.updateDescriptorSets(descriptorWrites, {});
    }
  }
  
  vk::raii::CommandBuffer beginSingleTimeCommands()
  {
    vk::CommandBufferAllocateInfo allocInfo{
      .commandPool = commandPool,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = 1
    };
    vk::raii::CommandBuffer commandBuffer = std::move(vk::raii::CommandBuffers(device, allocInfo).front());

    vk::CommandBufferBeginInfo beginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
    commandBuffer.begin(beginInfo);

    return std::move(commandBuffer);
  }

  void endSingleTimeCommands(vk::raii::CommandBuffer &&commandBuffer)
  {
    commandBuffer.end();

    vk::SubmitInfo submitInfo{
      .commandBufferCount = 1,
      .pCommandBuffers = &*commandBuffer
    };
    renderQueue.submit(submitInfo, nullptr);
    renderQueue.waitIdle();
  }
  
  void copyBuffer(vk::raii::Buffer &srcBuffer, vk::raii::Buffer &dstBuffer, vk::DeviceSize size)
  {
    vk::raii::CommandBuffer commandCopyBuffer = beginSingleTimeCommands();
    commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy{.size = size});
    endSingleTimeCommands(std::move(commandCopyBuffer));
  }
  
  uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)
  {
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
    
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
      if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
        return i;
      }
    }
    
    throw std::runtime_error("failed to find suitable memory type!");
  }
  
  void createCommandBuffers()
  {
    vk::CommandBufferAllocateInfo allocInfo {
      .commandPool = commandPool,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = MAX_FRAMES_IN_FLIGHT
    };
    commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
  }
  
  void recordCommandBuffer(uint32_t imageIndex)
  {
    auto &commandBuffer = commandBuffers[frameResourceIndex];
    commandBuffer.begin({});
    // Set up the color attachment then rendering info
    vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
    vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);

    if (deviceCapabilities.dynamicRenderingSupported) {
      // Before starting rendering, transition the swapchain image to vk::ImageLayout::eColorAttachmentOptimal
      transitionImageLayout(commandBuffer,
                            swapchainImages[imageIndex],
                            vk::ImageLayout::eUndefined,
                            vk::ImageLayout::eColorAttachmentOptimal,
                            {},                                             // no need to wait for previous operations
                            vk::AccessFlagBits2::eColorAttachmentWrite,
                            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                            vk::ImageAspectFlagBits::eColor,
                            0,
                            1);
      // Transition multisampled color image to vk::ImageLayout::eColorAttachmentOptimal
      transitionImageLayout(commandBuffer,
                            *colorImage,
                            vk::ImageLayout::eUndefined,
                            vk::ImageLayout::eColorAttachmentOptimal,
                            vk::AccessFlagBits2::eColorAttachmentWrite,
                            vk::AccessFlagBits2::eColorAttachmentWrite,
                            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                            vk::ImageAspectFlagBits::eColor,
                            0,
                            1);
      // Transition depth image to vk::ImageLayout::eDepthAttachmentOptimal
      transitionImageLayout(commandBuffer,
                            *depthImage,
                            vk::ImageLayout::eUndefined,
                            vk::ImageLayout::eDepthAttachmentOptimal,
                            vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                            vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                            vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                              vk::PipelineStageFlagBits2::eLateFragmentTests,
                            vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                              vk::PipelineStageFlagBits2::eLateFragmentTests,
                            vk::ImageAspectFlagBits::eDepth,
                            0,
                            1);
      // Color attachment (multisampled) with resolve attachment
      vk::RenderingAttachmentInfo colorAttachmentInfo {
        .imageView = colorImageView,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .resolveMode = vk::ResolveModeFlagBits::eAverage,
        .resolveImageView = swapchainImageViews[imageIndex],
        .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = clearColor
      };
      vk::RenderingAttachmentInfo depthAttachmentInfo {
        .imageView = depthImageView,
        .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
        .clearValue = clearDepth
      };
      vk::RenderingInfo renderingInfo {
        .renderArea = {.offset = {0, 0}, .extent = swapchainExtent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachmentInfo,
        .pDepthAttachment = &depthAttachmentInfo
      };
      commandBuffer.beginRendering(renderingInfo);
    } else {
      std::array<vk::ClearValue, 2> clearValues {clearColor, clearDepth};
      vk::RenderPassBeginInfo renderPassInfo {
        .renderPass = *renderPass,
        .framebuffer = *swapchainFramebuffers[imageIndex],
        .renderArea = {.offset = {0, 0}, .extent = swapchainExtent},
        .clearValueCount = static_cast<uint32_t>(clearValues.size()),
        .pClearValues = clearValues.data()
      };
      commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
    }

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
    commandBuffer.bindVertexBuffers(0, *vertexBuffer, {0});
    commandBuffer.bindIndexBuffer(*indexBuffer, 0, vk::IndexTypeValue<decltype(indices)::value_type>::value);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapchainExtent.width), static_cast<float>(swapchainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapchainExtent));
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, *descriptorSets[frameResourceIndex], nullptr);
    commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

    if (deviceCapabilities.dynamicRenderingSupported) {
      commandBuffer.endRendering();
      
      // After rendering, transition the swapchain image to vk::ImageLayout::ePresentSrcKHR
      transitionImageLayout(commandBuffer,
                            swapchainImages[imageIndex],
                            vk::ImageLayout::eColorAttachmentOptimal,
                            vk::ImageLayout::ePresentSrcKHR,
                            vk::AccessFlagBits2::eColorAttachmentWrite,
                            {},
                            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                            vk::PipelineStageFlagBits2::eBottomOfPipe,
                            vk::ImageAspectFlagBits::eColor,
                            0,
                            1);
    } else {
      commandBuffer.endRenderPass();
    }
    commandBuffer.end();
  }

  void transitionImageLayout(vk::raii::CommandBuffer &commandBuffer,
                             const vk::Image &image,
                             vk::ImageLayout oldLayout,
                             vk::ImageLayout newLayout,
                             vk::AccessFlags2 srcAccessMask,
                             vk::AccessFlags2 dstAccessMask,
                             vk::PipelineStageFlags2 srcStageMask,
                             vk::PipelineStageFlags2 dstStageMask,
                             vk::ImageAspectFlags imageAspectFlags,
                             uint32_t baseMipLevel,
                             uint32_t mipLevels)
  {
    if (deviceCapabilities.synchronization2Supported) {
      // Use commands in synchronization2 extension
      vk::ImageMemoryBarrier2 barrier {
        .srcStageMask = srcStageMask,
        .srcAccessMask = srcAccessMask,
        .dstStageMask = dstStageMask,
        .dstAccessMask = dstAccessMask,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {
          .aspectMask = imageAspectFlags,
          .baseMipLevel = baseMipLevel,
          .levelCount = mipLevels,
          .baseArrayLayer = 0,
          .layerCount = 1
        }
      };
      vk::DependencyInfo dependencyInfo {
        .dependencyFlags = {},
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier
      };
      commandBuffer.pipelineBarrier2(dependencyInfo);
    } else {
      // use compatibility synchronization commands
      vk::ImageMemoryBarrier barrier {
        .srcAccessMask = toLegacyAccessMask(srcAccessMask),
        .dstAccessMask = toLegacyAccessMask(dstAccessMask),
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {
          .aspectMask = imageAspectFlags,
          .baseMipLevel = baseMipLevel,
          .levelCount = mipLevels,
          .baseArrayLayer = 0,
          .layerCount = 1
        }
      };
      commandBuffer.pipelineBarrier(toLegacyStageMask(srcStageMask),
                                    toLegacyStageMask(dstStageMask),
                                    {},
                                    {},
                                    {},
                                    barrier);
    }
  }

  static vk::PipelineStageFlags toLegacyStageMask(vk::PipelineStageFlags2 stages)
  {
    vk::PipelineStageFlags legacyStages{};
    if (stages & vk::PipelineStageFlagBits2::eTopOfPipe) {
      legacyStages |= vk::PipelineStageFlagBits::eTopOfPipe;
    }
    if (stages & vk::PipelineStageFlagBits2::eColorAttachmentOutput) {
      legacyStages |= vk::PipelineStageFlagBits::eColorAttachmentOutput;
    }
    if (stages & vk::PipelineStageFlagBits2::eEarlyFragmentTests) {
      legacyStages |= vk::PipelineStageFlagBits::eEarlyFragmentTests;
    }
    if (stages & vk::PipelineStageFlagBits2::eLateFragmentTests) {
      legacyStages |= vk::PipelineStageFlagBits::eLateFragmentTests;
    }
    if (stages & vk::PipelineStageFlagBits2::eTransfer) {
      legacyStages |= vk::PipelineStageFlagBits::eTransfer;
    }
    if (stages & vk::PipelineStageFlagBits2::eFragmentShader) {
      legacyStages |= vk::PipelineStageFlagBits::eFragmentShader;
    }
    if (stages & vk::PipelineStageFlagBits2::eBottomOfPipe) {
      legacyStages |= vk::PipelineStageFlagBits::eBottomOfPipe;
    }
    if (stages & vk::PipelineStageFlagBits2::eAllCommands) {
      legacyStages |= vk::PipelineStageFlagBits::eAllCommands;
    }
    return legacyStages;
  }

  static vk::AccessFlags toLegacyAccessMask(vk::AccessFlags2 access)
  {
    vk::AccessFlags legacyAccess{};
    if (access & vk::AccessFlagBits2::eTransferRead) {
      legacyAccess |= vk::AccessFlagBits::eTransferRead;
    }
    if (access & vk::AccessFlagBits2::eTransferWrite) {
      legacyAccess |= vk::AccessFlagBits::eTransferWrite;
    }
    if (access & vk::AccessFlagBits2::eShaderRead) {
      legacyAccess |= vk::AccessFlagBits::eShaderRead;
    }
    if (access & vk::AccessFlagBits2::eColorAttachmentWrite) {
      legacyAccess |= vk::AccessFlagBits::eColorAttachmentWrite;
    }
    if (access & vk::AccessFlagBits2::eDepthStencilAttachmentWrite) {
      legacyAccess |= vk::AccessFlagBits::eDepthStencilAttachmentWrite;
    }
    return legacyAccess;
  }
  
  void createSyncObjects()
  {
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
    }
    if (deviceCapabilities.timelineSemaphoresSupported) {
      vk::SemaphoreTypeCreateInfo semaphoreType{
        .semaphoreType = vk::SemaphoreType::eTimeline,
        .initialValue = MAX_FRAMES_IN_FLIGHT
      };
      timelineSemaphore = vk::raii::Semaphore(device, {.pNext = &semaphoreType});
    } else {
      for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        inFlightFences.emplace_back(device, vk::FenceCreateInfo{
          .flags = vk::FenceCreateFlagBits::eSignaled
        });
      }
    }
    createRenderCompleteSemaphores();
  }

  void createRenderCompleteSemaphores()
  {
    renderCompleteSemaphores.reserve(swapchainImages.size());
    for (size_t i = 0; i < swapchainImages.size(); i++) {
      renderCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
    }
  }
  
  void updateUniformBuffer(uint32_t currentImage)
  {
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto  currentTime = std::chrono::high_resolution_clock::now();
    float time        = std::chrono::duration<float>(currentTime - startTime).count();

    UniformBufferObject ubo{};
    ubo.model = rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.view = lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.proj = glm::perspective(glm::radians(45.0f),
                                static_cast<float>(swapchainExtent.width) / static_cast<float>(swapchainExtent.height), 0.1f, 10.0f);
    ubo.proj[1][1] *= -1;

    memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
  }
  
  void drawFrame()
  {
    uint64_t timelineSignalValue = 0;
    if (deviceCapabilities.timelineSemaphoresSupported) {
      timelineSignalValue = frameIndex + MAX_FRAMES_IN_FLIGHT + 1;
      const uint64_t timelineWaitValue = timelineSignalValue - MAX_FRAMES_IN_FLIGHT;
      vk::SemaphoreWaitInfo waitInfo{
        .semaphoreCount = 1,
        .pSemaphores = &*timelineSemaphore,
        .pValues = &timelineWaitValue
      };
      if (device.waitSemaphores(waitInfo, UINT64_MAX) != vk::Result::eSuccess) {
        throw std::runtime_error("failed to wait for timeline semaphore!");
      }
    } else if (device.waitForFences(*inFlightFences[frameResourceIndex], vk::True, UINT64_MAX) != vk::Result::eSuccess) {
      throw std::runtime_error("failed to wait for in-flight fence!");
    }
    
    // acquire next image from swap chain
    auto [acquireResult, imageIndex] = swapchain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[frameResourceIndex], nullptr);
    
    // Due to VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS being defined, eErrorOutOfDateKHR can be checked as a result
    // here and does not need to be caught by an exception.
    if (acquireResult == vk::Result::eErrorOutOfDateKHR) {
      recreateSwapchain();
      return;
    } else if (acquireResult != vk::Result::eSuccess && acquireResult != vk::Result::eSuboptimalKHR) {
      // On other success codes than eSuccess and eSuboptimalKHR we just throw an exception.
      // On any error code, aquireNextImage already threw an exception.
      assert(acquireResult == vk::Result::eTimeout || acquireResult == vk::Result::eNotReady);
      throw std::runtime_error("failed to acquire swap chain image!");
    }
    
    updateUniformBuffer(frameResourceIndex);
    
    commandBuffers[frameResourceIndex].reset();
    recordCommandBuffer(imageIndex);

    if (deviceCapabilities.timelineSemaphoresSupported && deviceCapabilities.synchronization2Supported) {
      vk::SemaphoreSubmitInfo presentCompleteWaitInfo {
        .semaphore = *presentCompleteSemaphores[frameResourceIndex],
        .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput
      };
      vk::CommandBufferSubmitInfo commandBufferSubmitInfo {
        .commandBuffer = *commandBuffers[frameResourceIndex]
      };
      std::array<vk::SemaphoreSubmitInfo, 2> signalSemaphoreInfos {{
        {
          .semaphore = *renderCompleteSemaphores[imageIndex],
          .stageMask = vk::PipelineStageFlagBits2::eAllGraphics
        },
        {
          .semaphore = *timelineSemaphore,
          .value = timelineSignalValue,
          .stageMask = vk::PipelineStageFlagBits2::eAllCommands
        }
      }};
      vk::SubmitInfo2 submitInfo {
        .waitSemaphoreInfoCount = 1,
        .pWaitSemaphoreInfos = &presentCompleteWaitInfo,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &commandBufferSubmitInfo,
        .signalSemaphoreInfoCount = static_cast<uint32_t>(signalSemaphoreInfos.size()),
        .pSignalSemaphoreInfos = signalSemaphoreInfos.data()
      };
      renderQueue.submit2(submitInfo, nullptr);
    } else if (deviceCapabilities.timelineSemaphoresSupported) {
      std::array<vk::Semaphore, 2> signalSemaphores{
        *renderCompleteSemaphores[imageIndex],
        *timelineSemaphore
      };
      vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
      std::array<uint64_t, 1> waitValues{
        0 // The waited semaphore is binary, so its value is ignored.
      };
      std::array<uint64_t, 2> signalValues{
        0, // Binary semaphore value is ignored.
        timelineSignalValue
      };
      vk::TimelineSemaphoreSubmitInfo timelineSubmitInfo{
        .waitSemaphoreValueCount = static_cast<uint32_t>(waitValues.size()),
        .pWaitSemaphoreValues = waitValues.data(),
        .signalSemaphoreValueCount = static_cast<uint32_t>(signalValues.size()),
        .pSignalSemaphoreValues = signalValues.data()
      };
      vk::SubmitInfo submitInfo{
         .pNext = &timelineSubmitInfo,
         .waitSemaphoreCount = 1,
         .pWaitSemaphores = &*presentCompleteSemaphores[frameResourceIndex],
         .pWaitDstStageMask = &waitStage,
         .commandBufferCount = 1,
         .pCommandBuffers = &*commandBuffers[frameResourceIndex],
         .signalSemaphoreCount = static_cast<uint32_t>(signalSemaphores.size()),
         .pSignalSemaphores = signalSemaphores.data()
       };
      renderQueue.submit(submitInfo, nullptr);
    } else {
      device.resetFences(*inFlightFences[frameResourceIndex]);
      vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
      vk::SubmitInfo submitInfo {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*presentCompleteSemaphores[frameResourceIndex],
        .pWaitDstStageMask = &waitStage,
        .commandBufferCount = 1,
        .pCommandBuffers = &*commandBuffers[frameResourceIndex],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*renderCompleteSemaphores[imageIndex]
      };
      renderQueue.submit(submitInfo, *inFlightFences[frameResourceIndex]);
    }

    // Submit result back to swap chain to display on screen
    const vk::PresentInfoKHR presentInfoKHR {
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &*renderCompleteSemaphores[imageIndex],
      .swapchainCount = 1,
      .pSwapchains = &*swapchain,
      .pImageIndices = &imageIndex
    };
    auto presentResult = renderQueue.presentKHR(presentInfoKHR);
    // Due to VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS being defined, eErrorOutOfDateKHR can be checked as a result
    // here and does not need to be caught by an exception.
    if ((presentResult == vk::Result::eSuboptimalKHR) || (presentResult == vk::Result::eErrorOutOfDateKHR) || framebufferResized) {
      framebufferResized = false;
      recreateSwapchain();
    } else {
      // There are no other success codes than eSuccess; on any error code, presentKHR already threw an exception.
      assert(presentResult == vk::Result::eSuccess);
    }
    
    frameIndex++;
    frameResourceIndex = frameIndex % MAX_FRAMES_IN_FLIGHT;
  }
  
	[[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char> &code) const
	{
		vk::ShaderModuleCreateInfo createInfo{.codeSize = code.size() * sizeof(char), .pCode = reinterpret_cast<const uint32_t *>(code.data())};
		vk::raii::ShaderModule shaderModule{device, createInfo};

		return shaderModule;
	}
  
  static uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const &capabilities)
  {
    auto minImageCount = std::max(3u, capabilities.minImageCount);
    if ((0 < capabilities.maxImageCount) && (capabilities.maxImageCount < minImageCount)) {
      minImageCount = capabilities.maxImageCount;
    }
    return minImageCount;
  }

  static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const &availableFormats)
  {
    assert(!availableFormats.empty());
    const auto formatIt = std::ranges::find_if(
         availableFormats,
         [](const auto &format) {
           return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
    });
    return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
  }

  static vk::PresentModeKHR chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const &availablePresentModes)
  {
    assert(std::ranges::any_of(availablePresentModes, [](auto presentMode) { return presentMode == vk::PresentModeKHR::eFifo; }));
    return std::ranges::any_of(availablePresentModes,
                               [](const vk::PresentModeKHR value) { return vk::PresentModeKHR::eMailbox == value; }) ?
             vk::PresentModeKHR::eMailbox :
             vk::PresentModeKHR::eFifo;
  }

  vk::Extent2D chooseSwapExtent(vk::SurfaceCapabilitiesKHR const &capabilities)
  {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
      return capabilities.currentExtent;
    }
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    
    return {
      std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
      std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
    };
  }

  bool isDeviceSuitable(vk::raii::PhysicalDevice const &physicalDevice)
  {
    // Check if any of the queue families support graphics operations
    auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
    bool supportsGraphicsAndPresent =
      std::ranges::any_of(queueFamilyProperties,
          [&](auto const &qfp) {
            uint32_t qfIdx = static_cast<uint32_t>(&qfp - queueFamilyProperties.data());
            bool supportsGraphics = (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0);
            bool supportsPresent = physicalDevice.getSurfaceSupportKHR(qfIdx, surface);
            return supportsGraphics && supportsPresent;
          });

    // Check if all required physicalDevice extensions are available
    auto availableDeviceExtensions = physicalDevice.enumerateDeviceExtensionProperties();
    bool supportsAllRequiredExtensions =
      std::ranges::all_of(requiredDeviceExtension,
                          [&availableDeviceExtensions](auto const &requiredDeviceExtension) {
                            return std::ranges::any_of(availableDeviceExtensions,
                                                       [requiredDeviceExtension](auto const &availableDeviceExtension) { return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension) == 0; });
                          });

    // Check if physicalDevice supports required features (excluding those covered by compatibility checks)
    auto features = physicalDevice.template getFeatures2<vk::PhysicalDeviceFeatures2,
                                                         vk::PhysicalDeviceBufferDeviceAddressFeatures,
                                                         vk::PhysicalDeviceVulkan11Features,
                                                         vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT>();
    bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceFeatures2>().features.sampleRateShading &&
                                    features.template get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&
                                    features.template get<vk::PhysicalDeviceBufferDeviceAddressFeatures>().bufferDeviceAddress &&
                                    features.template get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters &&
                                    features.template get<vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT>().extendedDynamicState2;
    return supportsGraphicsAndPresent && supportsAllRequiredExtensions && supportsRequiredFeatures;
  }

  [[nodiscard]] std::vector<const char *> getRequiredInstanceExtensions() const
  {
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    
    // check if debug utils extension is available
    std::vector<vk::ExtensionProperties> props = context.enumerateInstanceExtensionProperties();
    bool debugUtilsAvailable = std::ranges::any_of(props,
                                                   [](vk::ExtensionProperties const &ep) { return
                                                       strcmp(ep.extensionName, vk::EXTDebugUtilsExtensionName) == 0;
                                                  });
    
    if (debugUtilsAvailable) {
      extensions.push_back(vk::EXTDebugUtilsExtensionName);
    } else {
      std::cout << "VK_EXT_debug_utils extension not available. Validation layers may not work." << std::endl;
    }
    
#ifdef USE_VULKAN_PORTABILITY_FEATURES
    extensions.push_back(vk::KHRPortabilityEnumerationExtensionName);
#endif
    return extensions;
  }
  
  static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                                                        vk::DebugUtilsMessageTypeFlagsEXT type,
                                                        const vk::DebugUtilsMessengerCallbackDataEXT *pCallbackData,
                                                        void *pUserData)
  {
    if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError || severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
    {
      std::cerr << "validation layer: type " << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;
    }

    return vk::False;
  }

  static std::vector<char> readFile(const std::string &fileName)
  {
    std::ifstream file(fileName, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + fileName);
    }
    std::vector<char> buffer(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    file.close();
    std::cout << "Successfully read file: " << fileName << " (" << buffer.size() << " bytes)" << std::endl;
    return buffer;
  }
};

int main()
{
	try
	{
		Application app;
		app.run();
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
