#pragma once
#include <cuda_runtime.h>

__device__ __host__ __forceinline__ unsigned DecodeMorton2X(unsigned code);
__device__ __host__ __forceinline__ unsigned DecodeMorton2Y(unsigned code);
__device__  __forceinline__ unsigned EncodeMorton2(unsigned x, unsigned y);
__host__  __forceinline__ unsigned EncodeMorton2h(unsigned x, unsigned y);

__host__ bool testreduct(unsigned seed = 0);


#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)
