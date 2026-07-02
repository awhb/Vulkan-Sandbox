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
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

const uint32_t WIDTH  = 800;
const uint32_t HEIGHT = 600;
const std::string MODEL_PATH = "models/viking_room.obj";
const std::string TEXTURE_PATH = "textures/viking_room.png";
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<char const *> validationLayers = {
  "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

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
};

struct UniformBufferObject
{
  alignas(16) glm::mat4 model;
  alignas(16) glm::mat4 view;
  alignas(16) glm::mat4 proj;
};

const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},

    {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}}};

const std::vector<uint16_t> indices = {
    0, 1, 2, 2, 3, 0,
    4, 5, 6, 6, 7, 4};

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
  vk::raii::Device device = nullptr;
  uint32_t renderQueueIdx = ~0;
  vk::raii::Queue renderQueue = nullptr;
  vk::raii::SwapchainKHR swapchain = nullptr;
  std::vector<vk::Image> swapchainImages;
  vk::SurfaceFormatKHR swapchainSurfaceFormat;
  vk::Extent2D swapchainExtent;
  std::vector<vk::raii::ImageView> swapchainImageViews;
  
  vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
  vk::raii::PipelineLayout pipelineLayout = nullptr;
  vk::raii::Pipeline graphicsPipeline = nullptr;
  
  vk::raii::Image depthImage = nullptr;
  vk::raii::DeviceMemory depthImageMemory = nullptr;
  vk::raii::ImageView depthImageView = nullptr;
  
  vk::raii::Image textureImage = nullptr;
  vk::raii::DeviceMemory textureImageMemory = nullptr;
  vk::raii::ImageView textureImageView = nullptr;
  vk::raii::Sampler textureSampler = nullptr;
  
  vk::raii::Buffer vertexBuffer = nullptr;
  vk::raii::DeviceMemory vertexBufferMemory = nullptr;
  vk::raii::Buffer indexBuffer = nullptr;
  vk::raii::DeviceMemory indexBufferMemory = nullptr;
  
  std::vector<vk::raii::Buffer> uniformBuffers;
  std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
  std::vector<void *> uniformBuffersMapped;
  
  vk::raii::DescriptorPool descriptorPool = nullptr;
  std::vector<vk::raii::DescriptorSet> descriptorSets;
  
  vk::raii::CommandPool commandPool = nullptr;
  std::vector<vk::raii::CommandBuffer> commandBuffers;
  
  std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
  std::vector<vk::raii::Semaphore> renderCompleteSemaphores;
  
  vk::raii::Semaphore timelineSemaphore = nullptr;
  uint32_t frameIndex = 0;
  uint32_t frameResourceIndex = 0;
  
  bool framebufferResized = false;
  
  std::vector<const char *> requiredDeviceExtension = {
    vk::KHRSwapchainExtensionName,
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
    createLogicalDevice();
    createSwapchain();
    createImageViews();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createCommandPool();
    createDepthResources();
    createTextureImage();
    createTextureSampler();
    createTextureImageView();
    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
    createSyncObjects();
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
    swapchainImageViews.clear();
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
    createDepthResources();
  }
  
  void createInstance()
  {
    constexpr vk::ApplicationInfo appInfo{
      .pApplicationName = "Hello Triangle",
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "No Engine",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = vk::ApiVersion14
    };
    
    // Get required validation layers
    std::vector<const char*> layers;
    if (enableValidationLayers) {
      layers.assign(validationLayers.begin(), validationLayers.end());
    }
    
    // Check if required layers are supported by Vulkan implementation
    auto layerProperties = context.enumerateInstanceLayerProperties();
    auto unsupportedLayerIt = std::ranges::find_if(
                                                   layers,
                                                   [&layerProperties](auto const &layer) {
                                                     return std::ranges::none_of(
                                                                                 layerProperties,
                                                                                 [layer](auto const &layerProperty) {
                                                                                   return strcmp(layerProperty.layerName, layer) == 0;
                                                                                 }
                                                                                 );
                                                   });
    if (unsupportedLayerIt != layers.end()) {
      throw std::runtime_error("Required validation layer not supported: " + std::string(*unsupportedLayerIt));
    }
    
    // Get required extensions
    auto extensions = getRequiredExtensions();
    
    // Check if required extensions are supported by Vulkan implementation
    auto extensionProperties = context.enumerateInstanceExtensionProperties();
    auto unsupportedExtensionIt = std::ranges::find_if(
                                                       extensions,
                                                       [&extensionProperties](auto const &extension) {
                                                         return std::ranges::none_of(
                                                                                     extensionProperties,
                                                                                     [extension](auto const &extensionProperty) {
                                                                                       return strcmp(extensionProperty.extensionName, extension) == 0;
                                                                                     }
                                                                                     );
                                                       }
                                                       );
    if (unsupportedExtensionIt != extensions.end()) {
      throw std::runtime_error("Required extension not supported: " + std::string(*unsupportedExtensionIt));
    }
    
    vk::InstanceCreateFlags instanceFlags{};
#ifdef USE_VULKAN_PORTABILITY_FEATURES
    instanceFlags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
#endif

    // Create instance info
    vk::InstanceCreateInfo createInfo{
      .flags = instanceFlags,
      .pApplicationInfo = &appInfo,
      .enabledLayerCount = static_cast<uint32_t>(layers.size()),
      .ppEnabledLayerNames = layers.data(),
      .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
      .ppEnabledExtensionNames = extensions.data()
    };

    instance = vk::raii::Instance(context, createInfo);
  }
  
  void setupDebugMessenger()
  {
    if (!enableValidationLayers) return;
    vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
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
    debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
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
    } else {
      throw std::runtime_error("failed to find a suitable GPU!");
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
    
    // query for Vulkan 1.3 features
    vk::StructureChain<vk::PhysicalDeviceFeatures2,
    vk::PhysicalDeviceVulkan11Features,
    vk::PhysicalDeviceVulkan13Features,
    vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
    vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR> featureChain = {
      {.features = {.samplerAnisotropy = true}},
      {.shaderDrawParameters = true},
      {.synchronization2 = true, .dynamicRendering = true},
      {.extendedDynamicState = true},
      {.timelineSemaphore = true}
    };
    
    // create a Device
    float queuePriority = 0.5f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo{
      .queueFamilyIndex = renderQueueIdx,
      .queueCount = 1,
      .pQueuePriorities = &queuePriority
    };
    vk::DeviceCreateInfo deviceCreateInfo{
      .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &deviceQueueCreateInfo,
      .enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtension.size()),
      .ppEnabledExtensionNames = requiredDeviceExtension.data()
    };
    
    device = vk::raii::Device(physicalDevice, deviceCreateInfo);
    renderQueue = vk::raii::Queue(device, renderQueueIdx, 0);
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
      swapchainImageViews.emplace_back(createImageView(image, swapchainSurfaceFormat.format, vk::ImageAspectFlagBits::eColor));
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
      .rasterizationSamples = vk::SampleCountFlagBits::e1,
      .sampleShadingEnable = vk::False
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
    
    vk::Format depthFormat = findDepthFormat();
    
    vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
      {
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
        .renderPass = nullptr // no traditional render pass given dynamic rendering
      },
      // This struct specifies the formats of the attachments used during (dynamic) rendering
      {
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &swapchainSurfaceFormat.format,
        .depthAttachmentFormat = depthFormat
      }
    };
    
    // second argument is pipeline cache (can store and reuse pipeline creation data to significantly speed up pipeline creation)
    graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
  }
  
  void createCommandPool()
  {
    vk::CommandPoolCreateInfo poolInfo {
      .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer, // allow command buffers to be rerecorded individually
      .queueFamilyIndex = renderQueueIdx
    };
    commandPool = vk::raii::CommandPool(device, poolInfo);
  }
  
  void createDepthResources()
  {
    vk::Format depthFormat = findDepthFormat();

    std::tie(depthImage, depthImageMemory) = createImage(swapchainExtent.width, swapchainExtent.height, depthFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal);
    depthImageView = createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth);
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
    int texWidth, texHeight, texChannels;
    stbi_uc *pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    vk::DeviceSize imageSize = texWidth * texHeight * 4;
    
    if (!pixels) {
      throw std::runtime_error("failed to load texture image!");
    }
    
    auto [stagingBuffer, stagingBufferMemory] = createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
                                                             vk::MemoryPropertyFlagBits::eHostVisible |
                                                             vk::MemoryPropertyFlagBits::eHostCoherent);
    void *data = stagingBufferMemory.mapMemory(0, imageSize);
    memcpy(data, pixels, imageSize);
    stagingBufferMemory.unmapMemory();
    
    stbi_image_free(pixels); // clean up original pixel array now
    
    std::tie(textureImage, textureImageMemory) = createImage(texWidth,
                                                             texHeight,
                                                             vk::Format::eR8G8B8A8Srgb,
                                                             vk::ImageTiling::eOptimal,
                                                             vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
                                                             vk::MemoryPropertyFlagBits::eDeviceLocal);
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
        vk::ImageAspectFlagBits::eColor
    );
    
    copyBufferToImage(commandBuffer, stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
    transitionImageLayout(
        commandBuffer,
        *textureImage,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eTransferWrite,
        vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eColor
    );
    endSingleTimeCommands(std::move(commandBuffer));
  }
  
  void createTextureImageView()
  {
    textureImageView = createImageView(*textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor);
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
      .compareOp = vk::CompareOp::eAlways
    };
    textureSampler = vk::raii::Sampler(device, samplerInfo);
  }
  
  vk::raii::ImageView createImageView(vk::Image const &image, vk::Format format, vk::ImageAspectFlags aspectFlags)
  {
    vk::ImageViewCreateInfo viewInfo {
      .image = image,
      .viewType = vk::ImageViewType::e2D,
      .format = format,
      .subresourceRange = {
        .aspectMask = aspectFlags,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1
      }
    };
    return vk::raii::ImageView(device, viewInfo);
  }
  
  std::pair<vk::raii::Image, vk::raii::DeviceMemory> createImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties)
  {
    vk::ImageCreateInfo imageInfo {
      .imageType = vk::ImageType::e2D,
      .format = format,
      .extent = {width, height, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = vk::SampleCountFlagBits::e1,
      .tiling = tiling,
      .usage = usage,
      .sharingMode = vk::SharingMode::eExclusive
    };
    
    vk::raii::Image image = vk::raii::Image(device, imageInfo);
    vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
    vk::MemoryAllocateInfo allocInfo {
      .allocationSize = memRequirements.size,
      .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
    };
    vk::raii::DeviceMemory imageMemory = vk::raii::DeviceMemory(device, allocInfo);
    image.bindMemory(*imageMemory, 0);
    return {std::move(image), std::move(imageMemory)};
  }
  
  void copyBufferToImage(vk::raii::CommandBuffer &commandBuffer, const vk::raii::Buffer &buffer, vk::raii::Image &image, uint32_t width, uint32_t height)
  {
    vk::BufferImageCopy region{.bufferOffset      = 0,
                               .bufferRowLength   = 0,
                               .bufferImageHeight = 0,
                               .imageSubresource  = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
                               .imageOffset       = {0, 0, 0},
                               .imageExtent       = {width, height, 1}};
    commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
  }
  
  std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                                                                   vk::MemoryPropertyFlags properties)
  {
    vk::BufferCreateInfo bufferInfo {.size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive};
    vk::raii::Buffer buffer = vk::raii::Buffer(device, bufferInfo);
    vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
    vk::MemoryAllocateInfo allocInfo {
      .allocationSize = memRequirements.size,
      .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
    };
    vk::raii::DeviceMemory bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
    buffer.bindMemory(*bufferMemory, 0); // associate buffer with device-allocated memory
    return {std::move(buffer), std::move(bufferMemory)};
  }
  
  void createVertexBuffer()
  {
    vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
    
    auto [stagingBuffer, stagingBufferMemory] = createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                                                             vk::MemoryPropertyFlagBits::eHostVisible |
                                                             vk::MemoryPropertyFlagBits::eHostCoherent);
    
    void *data = stagingBufferMemory.mapMemory(0, bufferSize);
    memcpy(data, vertices.data(), bufferSize);
    stagingBufferMemory.unmapMemory();
    
    std::tie(vertexBuffer, vertexBufferMemory) = createBuffer(bufferSize,
                                                              vk::BufferUsageFlagBits::eTransferDst |
                                                              vk::BufferUsageFlagBits::eVertexBuffer,
                                                              vk::MemoryPropertyFlagBits::eDeviceLocal);
    copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
  }
  
  void createIndexBuffer()
  {
    vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();
    
    auto [stagingBuffer, stagingBufferMemory] = createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                                                             vk::MemoryPropertyFlagBits::eHostVisible |
                                                             vk::MemoryPropertyFlagBits::eHostCoherent);
    void *data = stagingBufferMemory.mapMemory(0, bufferSize);
    memcpy(data, indices.data(), (size_t) bufferSize);
    stagingBufferMemory.unmapMemory();
    
    std::tie(indexBuffer, indexBufferMemory) = createBuffer(bufferSize,
                                                            vk::BufferUsageFlagBits::eTransferDst |
                                                            vk::BufferUsageFlagBits::eIndexBuffer,
                                                            vk::MemoryPropertyFlagBits::eDeviceLocal);
    copyBuffer(stagingBuffer, indexBuffer, bufferSize);
  }
  
  void createUniformBuffers()
  {
    vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      auto [buffer, bufferMem] = createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                                              vk::MemoryPropertyFlagBits::eHostVisible |
                                              vk::MemoryPropertyFlagBits::eHostCoherent);
      uniformBuffers.emplace_back(std::move(buffer));
      uniformBuffersMemory.emplace_back(std::move(bufferMem));
      uniformBuffersMapped.emplace_back(uniformBuffersMemory.back().mapMemory(0, bufferSize));
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
  
    // Before starting rendering, transition the swapchain image to vk::ImageLayout::eColorAttachmentOptimal
    transitionImageLayout(
        commandBuffer,
        swapchainImages[imageIndex],
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal,
        {},                                                        // srcAccessMask (no need to wait for previous operations)
        vk::AccessFlagBits2::eColorAttachmentWrite,                // dstAccessMask
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,        // srcStage
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,        // dstStage
        vk::ImageAspectFlagBits::eColor
    );
    
    // Transition depth image to vk::ImageLayout::eDepthAttachmentOptimal
    transitionImageLayout(
        commandBuffer,
        *depthImage,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eDepthAttachmentOptimal,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::ImageAspectFlagBits::eDepth
    );
    
    // Set up the color attachment then rendering info
    vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
    vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);
    
    vk::RenderingAttachmentInfo colorAttachmentInfo = {
      .imageView = swapchainImageViews[imageIndex],
      .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eStore,
      .clearValue = clearColor
    };
    
    vk::RenderingAttachmentInfo depthAttachmentInfo = {
      .imageView = depthImageView,
      .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eDontCare,
      .clearValue = clearDepth
    };
    
    vk::RenderingInfo renderingInfo = {
      .renderArea = {.offset = {0, 0}, .extent = swapchainExtent},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &colorAttachmentInfo,
      .pDepthAttachment = &depthAttachmentInfo
    };
    
    // rendering commands
    commandBuffer.beginRendering(renderingInfo);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
    commandBuffer.bindVertexBuffers(0, *vertexBuffer, {0});
    commandBuffer.bindIndexBuffer(*indexBuffer, 0, vk::IndexTypeValue<decltype(indices)::value_type>::value);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapchainExtent.width), static_cast<float>(swapchainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapchainExtent));
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, *descriptorSets[frameResourceIndex], nullptr);
    commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
    commandBuffer.endRendering();

    // After rendering, transition the swapchain image to vk::ImageLayout::ePresentSrcKHR
    transitionImageLayout(
        commandBuffer,
        swapchainImages[imageIndex],
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::ePresentSrcKHR,
        vk::AccessFlagBits2::eColorAttachmentWrite,                // srcAccessMask
        {},                                                        // dstAccessMask
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,        // srcStage
        vk::PipelineStageFlagBits2::eBottomOfPipe,                 // dstStage
        vk::ImageAspectFlagBits::eColor
    );
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
                             vk::ImageAspectFlags imageAspectFlags)
  {
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
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1
      }
    };
    
    vk::DependencyInfo dependency_info = {
      .dependencyFlags = {},
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &barrier
    };
    
    commandBuffer.pipelineBarrier2(dependency_info);
  }
  
  void createSyncObjects()
  {
    vk::SemaphoreTypeCreateInfo semaphoreType{
      .semaphoreType = vk::SemaphoreType::eTimeline,
      .initialValue = MAX_FRAMES_IN_FLIGHT
    };
    timelineSemaphore = vk::raii::Semaphore(device, {.pNext = &semaphoreType});
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
      presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
    }
    for (size_t i = 0; i < swapchainImages.size(); i++)
    {
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
    // Wait for timeline semaphore
    const uint64_t signalValue = frameIndex + MAX_FRAMES_IN_FLIGHT + 1;
    const uint64_t waitValue = signalValue - MAX_FRAMES_IN_FLIGHT;
    
    vk::SemaphoreWaitInfo waitInfo{
        .semaphoreCount = 1,
        .pSemaphores = &*timelineSemaphore,
        .pValues = &waitValue
    };
    auto waitResult = device.waitSemaphores(waitInfo, UINT64_MAX);
    
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
    
    vk::SemaphoreSubmitInfo presentCompleteWaitInfo {
      .semaphore = *presentCompleteSemaphores[frameResourceIndex],
      .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput
    };
    vk::CommandBufferSubmitInfo commandBufferSubmitInfo {
      .commandBuffer = *commandBuffers[frameResourceIndex],
    };
    std::vector<vk::SemaphoreSubmitInfo> signalSemaphoreInfos = {
      {
        .semaphore = *renderCompleteSemaphores[imageIndex],
        .stageMask = vk::PipelineStageFlagBits2::eAllGraphics
      },
      {
        .semaphore = *timelineSemaphore,
        .value = signalValue,
        .stageMask = vk::PipelineStageFlagBits2::eAllCommands
      }
    };
    
    const vk::SubmitInfo2 submitInfo {
      .waitSemaphoreInfoCount = 1,
      .pWaitSemaphoreInfos = &presentCompleteWaitInfo,
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos = &commandBufferSubmitInfo,
      .signalSemaphoreInfoCount = static_cast<uint32_t>(signalSemaphoreInfos.size()),
      .pSignalSemaphoreInfos = signalSemaphoreInfos.data()
    };
    renderQueue.submit2(submitInfo, nullptr);

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
    // Check if the physicalDevice supports the Vulkan 1.3 API version
    bool supportsVulkan1_3 = physicalDevice.getProperties().apiVersion >= vk::ApiVersion13;

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

    // Check if the physicalDevice supports the required features
    auto features = physicalDevice.template getFeatures2<vk::PhysicalDeviceFeatures2,
                                                         vk::PhysicalDeviceVulkan11Features,
                                                         vk::PhysicalDeviceVulkan13Features,
                                                         vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
                                                         vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>();
    bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&
                                    features.template get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters &&
                                    features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
                                    features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState &&
                                    features.template get<vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>().timelineSemaphore;

    // Return true if the physicalDevice meets all the criteria
    return supportsVulkan1_3 && supportsGraphicsAndPresent && supportsAllRequiredExtensions && supportsRequiredFeatures;
  }

  std::vector<const char*> getRequiredExtensions()
  {
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    
    if (enableValidationLayers) {
      extensions.push_back(vk::EXTDebugUtilsExtensionName);
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
