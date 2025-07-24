#pragma once
// top.cuh
#include "reduction.cuh"
#include "constmem.cuh"
#include <cuda_fp16.h>
// ----------------------------------------------------------------------------------------------
static __device__ __host__ __inline__
uint64_t get_top(const uint64_t src, __half2* fout /* [16] */, uint64_t& bot, uint64_t& mid){
	// 64
	constexpr uint64_t M = 0x1111'1111'1111'1111ULL;
	uint64_t sum0 = (src & M) + ((src >> 1) & M) + ((src >> 2) & M) + ((src >> 3) & M);
	sum0 >>= 1;			// 1 if 2 or 3
	sum0 |= sum0 >> 1;    // or if 4 ( res in pos 0)
	//sum0 &= M;
	bot = sum0 & M;

	// 16
	constexpr uint64_t M0 = 0x0001'0001'0001'0001ULL, M1 = M0 << 4, M2 = M1 << 4, M3 = M2 << 4;
	uint64_t sum1 = (sum0 & M0) + ((sum0 & M1) >> 4) + ((sum0 & M2) >> 8) + ((sum0 & M3) >> 12);
	sum1 >>= 1;			// 1 if 2 or 3
	sum1 |= sum1 >> 1;    // or if 4 ( res in pos 0)
	//sum1 &= M0;
	mid = sum1 & M0;

	// 4
	uint64_t sum2 = (sum1 & 0x0001) |
		((sum1 & 0x0001'0000) >> 15) |
		((sum1 & 0x0001'0000'0000) >> 30) |
		((sum1 & 0x0001'0000'0000'0000) >> 45);

	return sum2;
} 
// ----------------------------------------------------------------------------------------------
static __device__ __host__ __inline__
uint64_t get_top(const uint64_t src, __half2* fout /* [16] */){
	// 64
	constexpr uint64_t M = 0x1111'1111'1111'1111ULL;
	uint64_t sum0 = (src & M) + ((src >> 1) & M) + ((src >> 2) & M) + ((src >> 3) & M);
	sum0 >>= 1;			// 1 if 2 or 3
	sum0 |= sum0 >> 1;    // or if 4 ( res in pos 0)
	//sum0 &= M;	// TODO: del?

	// 16
	constexpr uint64_t M0 = 0x0001'0001'0001'0001ULL, M1 = M0 << 4, M2 = M1 << 4, M3 = M2 << 4;
	uint64_t sum1 = (sum0 & M0) + ((sum0 & M1) >> 4) + ((sum0 & M2) >> 8) + ((sum0 & M3) >> 12);
	sum1 >>= 1;			// 1 if 2 or 3
	sum1 |= sum1 >> 1;    // or if 4 ( res in pos 0)
	//sum1 &= M0;	// TODO: del?

	// 4
	uint64_t sum2 = (sum1 & 0x0001) |
		((sum1 & 0x0001'0000) >> 15) |
		((sum1 & 0x0001'0000'0000) >> 30) |
		((sum1 & 0x0001'0000'0000'0000) >> 45);
	return sum2;
}
// ----------------------------------------------------------------------------------------------
