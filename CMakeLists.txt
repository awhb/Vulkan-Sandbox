cmake_minimum_required (VERSION 3.29)

set (TEMPLATE_NAME "VulkanSandbox")

project (${TEMPLATE_NAME})

# Add option to enable/disable C++ 20 module
option(ENABLE_CPP20_MODULE "Enable C++ 20 module support for Vulkan" OFF)

# Add option to enable/disable Vulkan portability features
set(VULKAN_PORTABILITY_FEATURES_DEFAULT OFF)
if (APPLE)
  set(VULKAN_PORTABILITY_FEATURES_DEFAULT ON)
endif()
option(ENABLE_VULKAN_PORTABILITY_FEATURES "Enable Vulkan portability features for MoltenVK" ${VULKAN_PORTABILITY_FEATURES_DEFAULT})

# Enable C++ module dependency scanning only if C++ 20 module is enabled
if(ENABLE_CPP20_MODULE)
  set(CMAKE_CXX_SCAN_FOR_MODULES ON)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(Vulkan 1.4.335 REQUIRED)
find_package(tinyobjloader REQUIRED)
find_package(tinygltf REQUIRED)
find_package(KTX REQUIRED)

# Set up Vulkan C++ module only if enabled
if(ENABLE_CPP20_MODULE)
  add_library(VulkanCppModule)
  add_library(Vulkan::cppm ALIAS VulkanCppModule)

  target_compile_definitions(VulkanCppModule PUBLIC VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1)
  target_include_directories(VulkanCppModule PUBLIC "${Vulkan_INCLUDE_DIR}")
  target_link_libraries(VulkanCppModule PUBLIC Vulkan::Vulkan)

  set_target_properties(VulkanCppModule PROPERTIES CXX_STANDARD 20)

  # Add MSVC-specific compiler options for proper C++ module support
  if(MSVC)
    target_compile_options(VulkanCppModule PRIVATE
      /std:c++latest      # Use latest C++ standard for better module support
      /permissive-        # Standards conformance mode
      /Zc:__cplusplus     # Enable correct __cplusplus macro
      /EHsc               # Enable C++ exception handling
      /Zc:preprocessor    # Use conforming preprocessor
      /translateInclude   # Automatically translate #include to import for standard library
    )
  endif()

  target_sources(VulkanCppModule
    PUBLIC
    FILE_SET cxx_modules TYPE CXX_MODULES
    BASE_DIRS
    "${Vulkan_INCLUDE_DIR}"
    FILES
    "${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm"
  )


  # Add the vulkan.cppm file directly as a source file
  target_sources(VulkanCppModule
    PRIVATE
    "${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm"
  )
else()
  # Create a dummy interface library when C++ 20 module is disabled
  add_library(VulkanCppModule INTERFACE)
  add_library(Vulkan::cppm ALIAS VulkanCppModule)
  target_link_libraries(VulkanCppModule INTERFACE Vulkan::Vulkan)
  target_compile_definitions(VulkanCppModule
    INTERFACE VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
  )
endif()

find_package(stb REQUIRED)
set(STB_INCLUDEDIR ${stb_INCLUDE_DIRS})

find_program(SLANGC_EXECUTABLE slangc HINTS $ENV{VULKAN_SDK}/bin REQUIRED)

set(TEMPLATE_DIR ${CMAKE_BINARY_DIR}/${TEMPLATE_NAME})

set(SOURCES
  src/main.cpp
)
add_executable (${TEMPLATE_NAME})
target_sources(${TEMPLATE_NAME} PRIVATE ${SOURCES})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src/" FILES ${SOURCES})

set_target_properties (${TEMPLATE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TEMPLATE_DIR})
set_target_properties (${TEMPLATE_NAME} PROPERTIES CXX_STANDARD 20)
target_link_libraries (${TEMPLATE_NAME} Vulkan::cppm glfw)
target_include_directories (${TEMPLATE_NAME} PRIVATE ${STB_INCLUDEDIR})

# Add compile definition if C++ 20 module is enabled
if(ENABLE_CPP20_MODULE)
  target_compile_definitions(${TEMPLATE_NAME} PRIVATE USE_CPP20_MODULES=1)
endif()

# Add compile definition if Vulkan portability features are enabled
if(ENABLE_VULKAN_PORTABILITY_FEATURES)
  target_compile_definitions(${TEMPLATE_NAME} PRIVATE USE_VULKAN_PORTABILITY_FEATURES=1)
endif()

# Define VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS to treat VK_ERROR_OUT_OF_DATE_KHR as a success code
target_compile_definitions(${TEMPLATE_NAME} PRIVATE "VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS" )

# Set working directory for the executable to the template directory
if(WIN32)
  if(${CMAKE_GENERATOR} MATCHES "Visual Studio.*")
    set_target_properties(${TEMPLATE_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${TEMPLATE_DIR})
  endif()
elseif(APPLE)
  if (${CMAKE_GENERATOR} MATCHES "Xcode")
    set_target_properties(${TEMPLATE_NAME} PROPERTIES
      XCODE_GENERATE_SCHEME TRUE
      XCODE_SCHEME_WORKING_DIRECTORY "${TEMPLATE_DIR}"
    )
  endif()
endif()

# Slang shader
set (SHADER_TARGET Shaders)
set (SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set (SHADERS_DIR ${TEMPLATE_DIR}/shaders)
add_custom_command(OUTPUT ${SHADERS_DIR} COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR})
file(GLOB SHADER_SOURCES "${SHADER_SOURCE_DIR}/*.slang")

foreach(SHADER_SOURCE ${SHADER_SOURCES})
  get_filename_component(SHADER_NAME_WE "${SHADER_SOURCE}" NAME_WE)
  set(SHADER_SPV "${SHADERS_DIR}/${SHADER_NAME_WE}.spv")
  add_custom_command(
    OUTPUT  ${SHADER_SPV}
    COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCE} -target spirv -profile spirv_1_4 -emit-spirv-directly -o ${SHADER_SPV}
    WORKING_DIRECTORY ${SHADERS_DIR}
    DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
    COMMENT "Compiling Slang shader ${SHADER_NAME_WE}.slang"
    VERBATIM
  )
  list(APPEND SHADER_SPVS ${SHADER_SPV})
endforeach()
add_custom_target(${SHADER_TARGET} DEPENDS ${SHADER_SPVS} SOURCES ${SHADER_SOURCES})
add_dependencies(${TEMPLATE_NAME} ${SHADER_TARGET})
source_group(TREE "${SHADER_SOURCE_DIR}" FILES ${SHADER_SOURCES})

# Assets
set (ASSET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/assets/")
file (COPY ${ASSET_DIR}/viking_room.obj DESTINATION ${TEMPLATE_DIR}/models)
file (COPY ${ASSET_DIR}/viking_room.png DESTINATION ${TEMPLATE_DIR}/textures)
