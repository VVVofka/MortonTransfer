#pragma once
// top.cuh
#include "reduction.cuh"
#include "constmem.cuh"
#include <cuda_fp16.h>
// ----------------------------------------------------------------------------------------------
static __device__ __host__ __forceinline__ uint64_t pack4(uint64_t a){
	return (a & 0x0001) |
		((a & 0x0001'0000) >> 15) |
		((a & 0x0001'0000'0000) >> 30) |
		((a & 0x0001'0000'0000'0000) >> 45);
}// ----------------------------------------------------------------------------------------------
static __device__ __host__ __forceinline__ uint64_t pack16(uint64_t a){
	return (a & 0x1) | ((a & 0x10) >> 3) | ((a & 0x100) >> 6) | ((a & 0x1000) >> 9) |
		((a & 0x1'0000) >> 12) | ((a & 0x10'0000) >> 15) | ((a & 0x100'0000) >> 18) | ((a & 0x1000'0000) >> 21) |
		((a & 0x1'0000'0000) >> 24) | ((a & 0x10'0000'0000) >> 27) | ((a & 0x100'0000'0000) >> 30) | ((a & 0x1000'0000'0000) >> 33) |
		((a & 0x1'0000'0000'0000) >> 36) | ((a & 0x10'0000'0000'0000) >> 39) | ((a & 0x100'0000'0000'0000) >> 42) | ((a & 0x1000'0000'0000'0000) >> 45);
}// ----------------------------------------------------------------------------------------------
static __device__ __host__ __inline__ void get_topA4A16(const uint64_t src, uint64_t& out4, uint64_t& out16){
	// 64 -> 16 unpack
	constexpr uint64_t M = 0x1111'1111'1111'1111ULL;
	uint64_t sum0 = (src & M) + ((src >> 1) & M) + ((src >> 2) & M) + ((src >> 3) & M);
	sum0 >>= 1;			// 1 if 2 or 3
	sum0 |= sum0 >> 1;  // or if 4 ( res in pos 0)
	out16 = pack16(sum0);

	// 16 unpack -> 4 unpack
	constexpr uint64_t M0 = 0x0001'0001'0001'0001ULL, M1 = M0 << 4, M2 = M1 << 4, M3 = M2 << 4;
	uint64_t sum1 = (sum0 & M0) + ((sum0 & M1) >> 4) + ((sum0 & M2) >> 8) + ((sum0 & M3) >> 12);
	sum1 >>= 1;			// 1 if 2 or 3
	sum1 |= sum1 >> 1;  // or if 4 ( res in pos 0)
	out4 = pack4(sum1);
}// ===============================================================================================
// 32 thread: in_val[1] out[16]
static __global__ void glTop1(const uint64_t* __restrict__ in_val, __half2* __restrict__ out){
	__shared__ uint64_t a4, a16;
	__shared__ __half2 f0[2], f1[8];

	if(threadIdx.x == 0){
		get_topA4A16(in_val[0], a4, a16);
	}
	__syncthreads();	// __syncwarp();

	if(threadIdx.x < 2){
		const __half* pkf_1 = &kF4[a4 * 4];
		const __half2* pkf2_1 = reinterpret_cast<const __half2*>(pkf_1);
		f0[threadIdx.x] = pkf2_1[threadIdx.x] * kLay[0];	//if(threadIdx.x < 2)!
	}
	__syncthreads();	// __syncwarp();

	if(threadIdx.x < 8){
		const uint64_t a_2 = (a16 >> (4 * (threadIdx.x / 8))) & 0xF;	// (threadIdx.x / 8)=[0..3]
		const __half* pkf_2 = &kF4[a_2 * 4];	// size=4
		const __half2* pkf2_2 = reinterpret_cast<const __half2*>(pkf_2);	// size=2
		const __half2 kf = pkf2_2[(threadIdx.x >> 1) & 1];
		const __half fup = (reinterpret_cast<__half*>(f0))[threadIdx.x / 4];
		f1[threadIdx.x] = __half2half2(fup) + kf * kLay[1];
	}
	__syncthreads();	// __syncwarp();
	if((threadIdx.x & 1) == 0){
		const uint64_t a64 = in_val[0];
		const uint64_t a_3 = (a64 >> (4 * (threadIdx.x / 4))) & 0xF;	// (threadIdx.x / 2)=[0..15]
		const __half* pkf_3 = &kF4[a_3 * 4];
		const __half2* pkf2_3 = reinterpret_cast<const __half2*>(pkf_3);
		const __half2 kf = pkf2_3[(threadIdx.x >> 1) & 1];
		const __half fup = (reinterpret_cast<__half*>(f1))[threadIdx.x / 4];
		const __half2 res_3 = __half2half2(fup) + kf * kLay[2];
		out[threadIdx.x / 2] = res_3;
	}
}// ----------------------------------------------------------------------------------------------
template <unsigned COUNT_LAYS = 3>
static __global__ void glTop1N(const uint64_t* __restrict__ in_val, __half2* __restrict__ out){
	__shared__ uint64_t a4, a16;
	__shared__ __half2 f0[(COUNT_LAYS & 1) ? 1 << (COUNT_LAYS - 2) : 1 << COUNT_LAYS];
	__shared__ __half2 f1[(COUNT_LAYS & 1) ? 1 << COUNT_LAYS : 1 << (COUNT_LAYS - 2)];

#pragma unroll
	for(unsigned nlay = 0; nlay < COUNT_LAYS; nlay++){

	}

	if(threadIdx.x == 0){
		get_topA4A16(in_val[0], a4, a16);
	}
	__syncthreads();	// __syncwarp();

	if(threadIdx.x < 2){
		const __half* pkf_1 = &kF4[a4 * 4];
		const __half2* pkf2_2 = reinterpret_cast<const __half2*>(&pkf_1);
		f0[threadIdx.x] = pkf2_2[threadIdx.x] * kLay[0];	//if(threadIdx.x < 2)!
	}
	__syncthreads();	// __syncwarp();

	if(threadIdx.x < 8){
		const uint64_t a_2 = (a16 >> (4 * (threadIdx.x / 8))) & 0xF;	// (threadIdx.x / 8)=[0..3]
		const __half* pkf_2 = &kF4[a_2 * 4];
		const __half2* pkf2_2 = reinterpret_cast<const __half2*>(&pkf_2);
		f1[threadIdx.x] = f0[threadIdx.x / 16] + pkf2_2[threadIdx.x & 1] * kLay[1];
	}
	__syncthreads();	// __syncwarp();

	const uint64_t a64 = in_val[0];
	const uint64_t a_3 = (a64 >> (4 * (threadIdx.x / 2))) & 0xF;	// (threadIdx.x / 2)=[0..15]
	const __half* pkf_3 = &kF4[a_3 * 4];
	const __half2* pkf2_3 = reinterpret_cast<const __half2*>(&pkf_3);
	const __half2 res_3 = f1[threadIdx.x / 4] + pkf2_3[threadIdx.x & 1] * kLay[2];

	out[threadIdx.x] = res_3;
}// ----------------------------------------------------------------------------------------------
