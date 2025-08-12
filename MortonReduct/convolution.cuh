#pragma once
// convolution.cuh
#include <cuda_runtime.h>

static __host__ __device__ __forceinline__
uint64_t convolution64to16(const uint64_t src){
	constexpr uint64_t M = 0x1111'1111'1111'1111ULL;
	uint64_t sum = (src & M) + ((src >> 1) & M) + ((src >> 2) & M) + ((src >> 3) & M);
	// 1 if >=2 else 0
	sum >>= 1;  // 1 if 2 or 3
	sum |= sum >> 1;    // or 1 if 4 ( res in pos 0)
	return sum;
} // -----------------------------------------------------------------------------------

static __host__ __device__ __forceinline__
uint32_t convolution32to8(const uint32_t src){
	constexpr uint32_t M = 0x1111'1111;
	uint32_t sum = (src & M) + ((src >> 1) & M) + ((src >> 2) & M) + ((src >> 3) & M);
	// 1 if >=2 else 0
	sum >>= 1;  // 1 if 2 or 3
	sum |= sum >> 1;    // or 1 if 4 ( res in pos 0)
	return sum;
} // -----------------------------------------------------------------------------------