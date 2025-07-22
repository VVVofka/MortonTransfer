#pragma once
// top.cuh
#include "reduction.cuh"
#include "constmem.cuh"
#include <cuda_fp16.h>

static __device__ __inline__ 
void get_top(const uint64_t src, __half2* fout /* [16] */){
	// 64
	constexpr uint64_t M = 0x1111'1111'1111'1111ULL;
	uint64_t sum0 = (src & M) + ((src >> 1) & M) + ((src >> 2) & M) + ((src >> 3) & M);
	sum0 >>= 1;			// 1 if 2 or 3
	sum0 |= sum0 >> 1;    // or if 4 ( res in pos 0)

	// 16
	constexpr uint64_t M1 = 0x0001'0001'0001'0001ULL;
	uint64_t sum1 = (sum0 & M1) + ((sum0 >> 4) & M1) + ((sum0 >> 8) & M1) + ((sum0 >> 12) & M1);
	sum1 >>= 1;			// 1 if 2 or 3
	sum1 |= sum1 >> 1;    // or if 4 ( res in pos 0)

	// 4
	uint64_t sum2 = (sum1 & 0x0001) |
		((sum1 & 0x0001'0000) >> 15) |
		((sum1 & 0x0001'0000'0000) >> 30) |
		((sum1 & 0x0001'0000'0000'0000) >> 45);


}