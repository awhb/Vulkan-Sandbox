# FindVMA.cmake
#
# Finds Vulkan Memory Allocator and VulkanMemoryAllocator-Hpp.
#
# This will define the following variables:
#
#   VMA_FOUND             - True if both VMA and VMA-Hpp are available
#   VMA_INCLUDE_DIRS      - Include directories for vk_mem_alloc.h
#   VMA_HPP_FOUND         - True if vk_mem_alloc.hpp is available
#   VMA_HPP_INCLUDE_DIRS  - Include directories for vk_mem_alloc.hpp
#
# and the following imported targets:
#
#   VMA::VMA
#   VMA::VMAHpp
#

include(FindPackageHandleStandardArgs)

# Keep these together: VMA-Hpp tracks a compatible VMA version and its release
# layout includes the matching vk_mem_alloc.h.
set(_VMA_ROOT_HINTS
  "$ENV{VULKAN_SDK}/include"
  "${CMAKE_SOURCE_DIR}/external"
  "${CMAKE_SOURCE_DIR}/third_party"
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../external"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../third_party"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../attachments/external"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../attachments/third_party"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../attachments/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../external"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../third_party"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../include"
)

set(_VMA_PATH_SUFFIXES
  include
  VulkanMemoryAllocator
  VulkanMemoryAllocator/include
  VulkanMemoryAllocator-Hpp
  VulkanMemoryAllocator-Hpp/include
  vma
  vma/include
)

function(_vma_find_first_target out_var)
  foreach(candidate IN LISTS ARGN)
    if(TARGET ${candidate})
      set(${out_var} ${candidate} PARENT_SCOPE)
      return()
    endif()
  endforeach()

  set(${out_var} "" PARENT_SCOPE)
endfunction()

find_package(VulkanMemoryAllocator CONFIG QUIET)
find_package(VulkanMemoryAllocator-Hpp CONFIG QUIET)

_vma_find_first_target(_VMA_PACKAGE_TARGET
  GPUOpen::VulkanMemoryAllocator
  VulkanMemoryAllocator::VulkanMemoryAllocator
  VulkanMemoryAllocator::Headers
  VulkanMemoryAllocator
)

_vma_find_first_target(_VMA_HPP_PACKAGE_TARGET
  VulkanMemoryAllocator-Hpp::VulkanMemoryAllocator-Hpp
  VulkanMemoryAllocator-Hpp
)

find_path(VMA_INCLUDE_DIR
  NAMES vk_mem_alloc.h
  PATHS ${_VMA_ROOT_HINTS}
  PATH_SUFFIXES ${_VMA_PATH_SUFFIXES}
)

find_path(VMA_HPP_INCLUDE_DIR
  NAMES vk_mem_alloc.hpp
  PATHS ${_VMA_ROOT_HINTS}
  PATH_SUFFIXES ${_VMA_PATH_SUFFIXES}
)

if(NOT VMA_INCLUDE_DIR AND VMA_HPP_INCLUDE_DIR AND EXISTS "${VMA_HPP_INCLUDE_DIR}/vk_mem_alloc.h")
  set(VMA_INCLUDE_DIR "${VMA_HPP_INCLUDE_DIR}" CACHE PATH "Vulkan Memory Allocator include directory" FORCE)
endif()

if((NOT _VMA_PACKAGE_TARGET AND NOT VMA_INCLUDE_DIR) OR
   (NOT _VMA_HPP_PACKAGE_TARGET AND NOT VMA_HPP_INCLUDE_DIR))
  include(FetchContent)

  message(STATUS "VMA or VMA-Hpp not found, fetching VulkanMemoryAllocator-Hpp from GitHub...")

  if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
  endif()

  FetchContent_Declare(
    vmahpp
    GIT_REPOSITORY https://github.com/YaaZ/VulkanMemoryAllocator-Hpp.git
    GIT_TAG v3.3.0+3
    GIT_SHALLOW TRUE
    GIT_SUBMODULES_RECURSE TRUE
  )

  FetchContent_GetProperties(vmahpp)
  if(NOT vmahpp_POPULATED)
    FetchContent_Populate(vmahpp)
  endif()

  if(NOT VMA_INCLUDE_DIR AND EXISTS "${vmahpp_SOURCE_DIR}/VulkanMemoryAllocator/include/vk_mem_alloc.h")
    set(VMA_INCLUDE_DIR "${vmahpp_SOURCE_DIR}/VulkanMemoryAllocator/include" CACHE PATH "Vulkan Memory Allocator include directory" FORCE)
  endif()

  if(NOT VMA_HPP_INCLUDE_DIR AND EXISTS "${vmahpp_SOURCE_DIR}/include/vk_mem_alloc.hpp")
    set(VMA_HPP_INCLUDE_DIR "${vmahpp_SOURCE_DIR}/include" CACHE PATH "VulkanMemoryAllocator-Hpp include directory" FORCE)
  endif()

  if(NOT VMA_INCLUDE_DIR AND VMA_HPP_INCLUDE_DIR AND EXISTS "${VMA_HPP_INCLUDE_DIR}/vk_mem_alloc.h")
    set(VMA_INCLUDE_DIR "${VMA_HPP_INCLUDE_DIR}" CACHE PATH "Vulkan Memory Allocator include directory" FORCE)
  endif()
endif()

set(VMA_INCLUDE_DIRS ${VMA_INCLUDE_DIR})
set(VMA_HPP_INCLUDE_DIRS ${VMA_HPP_INCLUDE_DIR})

set(_VMA_AVAILABLE FALSE)
if(_VMA_PACKAGE_TARGET OR VMA_INCLUDE_DIR)
  set(_VMA_AVAILABLE TRUE)
endif()

set(VMA_HPP_FOUND FALSE)
if(_VMA_HPP_PACKAGE_TARGET OR VMA_HPP_INCLUDE_DIR)
  set(VMA_HPP_FOUND TRUE)
endif()

find_package_handle_standard_args(VMA
  REQUIRED_VARS _VMA_AVAILABLE VMA_HPP_FOUND
  FAIL_MESSAGE "Could not find Vulkan Memory Allocator and VulkanMemoryAllocator-Hpp"
)

if(VMA_FOUND AND NOT TARGET VMA::VMA)
  add_library(VMA::VMA INTERFACE IMPORTED)

  if(VMA_INCLUDE_DIRS)
    set_target_properties(VMA::VMA PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${VMA_INCLUDE_DIRS}"
    )
  endif()

  if(_VMA_PACKAGE_TARGET)
    set_property(TARGET VMA::VMA APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES ${_VMA_PACKAGE_TARGET}
    )
  endif()
endif()

if(VMA_FOUND AND VMA_HPP_FOUND AND NOT TARGET VMA::VMAHpp)
  add_library(VMA::VMAHpp INTERFACE IMPORTED)

  if(VMA_HPP_INCLUDE_DIRS)
    set_target_properties(VMA::VMAHpp PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${VMA_HPP_INCLUDE_DIRS}"
    )
  endif()

  if(_VMA_HPP_PACKAGE_TARGET)
    set_property(TARGET VMA::VMAHpp APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES ${_VMA_HPP_PACKAGE_TARGET}
    )
  endif()

  set_property(TARGET VMA::VMAHpp APPEND PROPERTY
    INTERFACE_LINK_LIBRARIES VMA::VMA
  )
endif()

mark_as_advanced(VMA_INCLUDE_DIR VMA_HPP_INCLUDE_DIR)
