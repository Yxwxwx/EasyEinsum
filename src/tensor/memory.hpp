#pragma once
#ifndef EASY_EINSUM_MEMORY_HEADER
#define EASY_EINSUM_MEMORY_HEADER

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#ifdef _USE_TBB_
#include <tbb/scalable_allocator.h>
#endif

#ifdef _USE_CUDA_
#include <cuda_runtime.h>
#endif

namespace EasyEinsum {

enum class Device { CPU, CUDA };

// default alignment
static constexpr size_t DEFAULT_ALIGNMENT = alignof(std::max_align_t);

/**
 * @brief Allocate memory on the specified device
 * @tparam Dev: target device (CPU or CUDA)
 */
template <Device Dev>
[[nodiscard]] inline void* mem_malloc(size_t size) {
  if (size == 0) return nullptr;
  void* p = nullptr;

  if constexpr (Dev == Device::CPU) {
#ifdef _USE_TBB_
    p = scalable_malloc(size);
#else
    p = std::malloc(size);
#endif
  } else {
#ifdef _USE_CUDA_
    cudaError_t err = cudaMalloc(&p, size);
    // cudaError_t err = cudaMallocManaged(&p, size);
    if (err != cudaSuccess) p = nullptr;
#else
    throw std::runtime_error(
        "CUDA support not enabled (_USE_CUDA_ not defined)");
#endif
  }

#if _DEBUG
  if (!p) {
    std::fprintf(stderr, "EasyEinsum::mem_malloc OOM (%zu bytes) on %s\n", size,
                 (Dev == Device::CPU ? "CPU" : "CUDA"));
    std::abort();
  }
#endif
  return p;
}

/**
 * @brief Allocate memory on the specified device and initialize it to zero
 * @tparam Dev: target device (CPU or CUDA)
 */
template <Device Dev>
[[nodiscard]] inline void* mem_calloc(size_t num, size_t size) {
  size_t total = num * size;
  if (total == 0) return nullptr;

  if constexpr (Dev == Device::CPU) {
#ifdef _USE_TBB_
    void* p = scalable_calloc(num, size);
#else
    void* p = std::calloc(num, size);
#endif
    return p;
  } else {
#ifdef _USE_CUDA_
    void* p = mem_malloc<Device::CUDA>(total);
    if (p) cudaMemset(p, 0, total);
    return p;
#else
    throw std::runtime_error("CUDA support not enabled");
#endif
  }
}

/**
 * @brief Free memory on the specified device
 * @tparam Dev: target device (CPU or CUDA)
 */
template <Device Dev>
inline void mem_free(void* ptr) noexcept {
  if (!ptr) return;

  if constexpr (Dev == Device::CPU) {
#ifdef _USE_TBB_
    scalable_free(ptr);
#else
    std::free(ptr);
#endif
  } else {
#ifdef _USE_CUDA_
    cudaFree(ptr);
#endif
  }
}

}  // namespace EasyEinsum

#endif