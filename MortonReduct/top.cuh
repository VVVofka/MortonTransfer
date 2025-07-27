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


// 32 thread: in_val[1](64bit) out[32](half2)
static __global__ void glTop1(const uint64_t* __restrict__ in_val, __half2* __restrict__ out){
	__shared__ uint64_t src, a4, a16;
	__shared__ __half2 f0[2], f1[8];

	if(threadIdx.x == 0){
		src = in_val[0];
		get_topA4A16(src, a4, a16); // a4: 4 bit used only; a16: 16 bit used only;
	}
	__syncthreads();

	if(threadIdx.x < 2){
		const __half* pkf_1 = &kF4[a4 * 4];
		const __half2* pkf2_1 = reinterpret_cast<const __half2*>(pkf_1);
		f0[threadIdx.x] = pkf2_1[threadIdx.x] * kLay[0];
	}
	__syncthreads();

	if(threadIdx.x < 8){
		const uint64_t a_2 = (a16 >> (4 * (threadIdx.x / 2))) & 0xF;
		const __half* pkf_2 = &kF4[a_2 * 4];	// size=4
		const __half2* pkf2_2 = reinterpret_cast<const __half2*>(pkf_2);	// size=2
		const __half2 kf = pkf2_2[threadIdx.x & 1];
		const __half fup = (reinterpret_cast<__half*>(f0))[threadIdx.x / 2];
		f1[threadIdx.x] = __half2half2(fup) + kf * kLay[1];
	}
	__syncthreads();
	const uint64_t a64 = src;	// 64
	const uint64_t a_3 = (a64 >> (4 * (threadIdx.x / 2))) & 0xF;
	const __half* pkf_3 = &kF4[a_3 * 4];	// size=4
	const __half2* pkf2_3 = reinterpret_cast<const __half2*>(pkf_3);	// size=2
	const __half2 kf = pkf2_3[threadIdx.x & 1];
	const __half fup = (reinterpret_cast<__half*>(f1))[threadIdx.x / 2];
	const __half2 res_3 = __half2half2(fup) + kf * kLay[2];
	out[threadIdx.x] = res_3;
}// ----------------------------------------------------------------------------------------------

// 128 thread: in_val[4](4*64=256bit) out[128](half2)
static __global__ void glTop2(const uint64_t* __restrict__ in_val, __half2* __restrict__ out){
	__shared__ uint64_t src[4], a4, a16, a64;
	__shared__ __half2 f0[2], f1[8], f2[32];
	if(threadIdx.x < 4){
		src[threadIdx.x] = in_val[threadIdx.x];
		if(threadIdx.x == 0){
			a64 = reduct64by1bit(src);
			get_topA4A16(a64, a4, a16); // a4: 4 bit used only; a16: 16 bit used only;
		}
	}
	__syncthreads();

	if(threadIdx.x < 2){
		const __half* pkf_1 = &kF4[a4 * 4];
		const __half2* pkf2_1 = reinterpret_cast<const __half2*>(pkf_1);
		f0[threadIdx.x] = pkf2_1[threadIdx.x] * kLay[0];
	}
	__syncthreads();

	if(threadIdx.x < 8){
		const uint64_t a_2 = (a16 >> (4 * (threadIdx.x / 2))) & 0xF;
		const __half* pkf_2 = &kF4[a_2 * 4];	// size=4
		const __half2* pkf2_2 = reinterpret_cast<const __half2*>(pkf_2);	// size=2
		const __half2 kf = pkf2_2[threadIdx.x & 1];
		const __half fup = (reinterpret_cast<__half*>(f0))[threadIdx.x / 2];
		f1[threadIdx.x] = __half2half2(fup) + kf * kLay[1];
	}
	__syncthreads();

	if(threadIdx.x < 32){
		const uint64_t a_3 = (a64 >> (4 * (threadIdx.x / 2))) & 0xF;
		const __half* pkf_3 = &kF4[a_3 * 4];	// size=4
		const __half2* pkf2_3 = reinterpret_cast<const __half2*>(pkf_3);	// size=2
		const __half2 kf = pkf2_3[threadIdx.x & 1];
		const __half fup = (reinterpret_cast<__half*>(f1))[threadIdx.x / 2];
		f2[threadIdx.x] = __half2half2(fup) + kf * kLay[2];
	}
	__syncthreads();	// threadIdx.x < 128
	const uint64_t a_4 = (src[threadIdx.x / 32] >> (4 * ((threadIdx.x & 31) / 2))) & 0xF;
	const __half* pkf_4 = &kF4[a_4 * 4];	// size=4
	const __half2* pkf2_4 = reinterpret_cast<const __half2*>(pkf_4);	// size=2
	const __half2 kf = pkf2_4[threadIdx.x & 1];
	const __half fup = (reinterpret_cast<__half*>(f2))[threadIdx.x / 2];
	const __half2 res_4 = __half2half2(fup) + kf * kLay[3];

	out[threadIdx.x] = res_4;
}// ----------------------------------------------------------------------------------------------
