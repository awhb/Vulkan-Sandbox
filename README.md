# About
Vulkan rendering engine prototype made with `Vulkan 1.4`, meant as an iterative experimental sandbox to learn best practices for Vulkan, engine architecture and low-level graphics programming. Code is forked with significant changes from a [Khronos Group Vulkan tutorial](https://github.com/KhronosGroup/Vulkan-Tutorial). Project is written for the `C++20` standard and is meant to be cross-platform. Currently, the code is tested only on `macOS`, using `XCode` generator but includes tutorial-sourced code that caters to `Windows` and `Linux` systems.

## Current Features
* Window handling with with [glfw](https://github.com/glfw/glfw)
* Vulkan-Hpp RAII bindings for Vulkan resource management
* GPU memory allocator with [VulkanMemoryAllocator-Hpp](https://github.com/YaaZ/VulkanMemoryAllocator-Hpp)
* Validation layer support in debug builds
* Dynamic rendering instead of render passes and framebuffers
* Timeline semaphores for GPU-CPU synchronization instead of fences
* Multiple frames in flight
* Slang shader source compiled to SPIR-V as part of CMake build
* CMake options for Vulkan portability support on MoltenVK/macOS and optional Vulkan C++20 module usage
* Mesh loading with [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader)

## Installation
This project uses [CMake](https://cmake.org/download/) as a build tool. Since the project is built using `Vulkan`, the latest [Vulkan SDK](https://vulkan.lunarg.com) is required.

## Requirements
Make sure that your graphics card can support listed Vulkan features and make sure you have updated graphics card driver.

## License
Distributed under the MIT License. See `LICENSE` for more information.
