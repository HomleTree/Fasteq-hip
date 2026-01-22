#ifndef WARP_PRIMITIVES_HPP
#define WARP_PRIMITIVES_HPP

#ifdef __HIP_PLATFORM_AMD__

// 异步内存操作映射
#define cudaMallocAsync hipMallocAsync
#define cudaFreeAsync hipFreeAsync

// CUB → hipCUB 映射
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;

#else
#include <cub/cub.cuh>
#endif

// 核心修复：自动映射 CUDA 类型到 HIP
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>

// HIP 掩码类型是 uint64_t，但兼容 CUDA 代码需要映射到 unsigned int
// 警告：这会丢失高 32 位，但 warp 原语在 CUDA 中只使用低 32 位
#define WARP_SIZE 64
#define WARP_MASK 0xffffffffU  // 改为 32 位，避免类型转换警告

// CUDA→HIP 类型映射
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString

#elif defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC__)
#include <cuda_runtime.h>
#define WARP_SIZE 32
#define WARP_MASK 0xffffffffU
#endif

// 前向声明（模板函数必须在实现前声明）
template<typename T>
__device__ __forceinline__ T __shfl_down_sync(unsigned int mask, T value, int offset);

template<typename T>
__device__ __forceinline__ T __shfl_sync(unsigned int mask, T value, int lane_id);

// 实现部分
#ifdef __HIP_PLATFORM_AMD__
#define __activemask() (WARP_MASK)

template<typename T>
__device__ __forceinline__ T __shfl_down_sync(unsigned int, T value, int offset) {
    return __shfl_down(value, offset, WARP_SIZE);
}

template<typename T>
__device__ __forceinline__ T __shfl_sync(unsigned int, T value, int lane_id) {
    return __shfl(value, lane_id, WARP_SIZE);
}

#elif defined(__HIP_PLATFORM_NVIDIA__)
#define __activemask() (WARP_MASK)
// CUDA 原生支持
#endif

#endif
