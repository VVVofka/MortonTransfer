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
} // ----------------------------------------------------------------------------------------------
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
} // ----------------------------------------------------------------------------------------------
__device__ __forceinline__ __half2 FourToOneTop(const uint64_t* src, const __half2* f_prev, const __half2 klay){
	const uint64_t a_5 = (src[threadIdx.x / 32] >> (4 * ((threadIdx.x & 31) / 2))) & 0xF;
	const __half2 kf = reinterpret_cast<const __half2*>(kF4 + a_5 * 4)[threadIdx.x & 1];
	const __half fup = (reinterpret_cast<const __half*>(f_prev))[threadIdx.x / 2];
	return __half2half2(fup) + kf * klay;
} // ----------------------------------------------------------------------------------------------
__device__ __forceinline__ __half2 FourToOneTop(const uint64_t src, const __half2* f_prev, const __half2 klay){
	const uint64_t a_3 = (src >> (4 * (threadIdx.x / 2))) & 0xF;	// wo (threadIdx.x & 31)
	const __half2 kf = reinterpret_cast<const __half2*>(kF4 + a_3 * 4)[threadIdx.x & 1];
	const __half fup = (reinterpret_cast<const __half*>(f_prev))[threadIdx.x / 2];
	return  __half2half2(fup) + kf * klay;
} // ===============================================================================================
// GridDim.x = 512 blocks; BlockDim.x = 1024 threads;
// Lay4: half2[512] f_up; (1'024 half) side=32
// Lay5: half2[2'048] f_up; (4'096 half) side=64
// Lay6: half2[8'192] f_up; (16'384 half) side=128
// Lay7: half2[32'768] f_up; (65'536 half) side=256
// Lay8: half2[131'072] f_up; (262'144 half) side=512
// Lay9: half2[524'288] f_up; (1'048'576 half) side=1024
// Parametrs:
// half2 f_up_in[512] =BlockDim; 1024 values;
// uint64_t a_dn_in[16384] =id/32 (32 per block) - recalculate all 'a' for avoid global memory access.
// half2 fout[524'288] //see Lay9 (1024 per block; 1 per thread)
static __global__ void glDnMid1(const __half2* __restrict__ f_up_in,
							const uint64_t* __restrict__ a_dn_in,
							__half2* __restrict__ f_out){
	const unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
	const uint64_t a_5 = (a_dn_in[id / 32] >> (4 * ((threadIdx.x & 31) / 2))) & 0xF;
	const __half2 kf = reinterpret_cast<const __half2*>(kF4 + a_5 * 4)[threadIdx.x & 1];
	const __half fup = (reinterpret_cast<const __half*>(f_up_in))[id / 2];
	f_out[id] = __half2half2(fup) + kf * klay;
}// ===============================================================================================
static __global__ void glDnMid3(const __half2* __restrict__ f_up_in, const uint64_t* in_val,
							__half2* __restrict__ f_out, const __half2 klay){
	__shared__ uint64_t src[16], a4, a16, a64, a256[4];
	__shared__ __half2 f0[2], f1[8], f2[32], f3[128];

	if(threadIdx.x < 16)
		src[threadIdx.x] = in_val[threadIdx.x];
	__syncwarp();

	if(threadIdx.x < 4)
		a256[threadIdx.x] = reduct64by1bit(src + threadIdx.x * 4);
	__syncwarp();

	if(threadIdx.x == 0){
		a64 = reduct64by1bit(a256);
		get_topA4A16(a64, a4, a16); // a4: 4 bit used only; a16: 16 bit used only;
	}
	__syncthreads();

	if(threadIdx.x < 2){
		const __half* pkf_1 = &kF4[a4 * 4];
		const __half2* pkf2_1 = reinterpret_cast<const __half2*>(pkf_1);
		f0[threadIdx.x] = pkf2_1[threadIdx.x] * kLay[0];
	}
	__syncthreads();

	if(threadIdx.x < 8)
		f1[threadIdx.x] = FourToOneTop(a16, f0, kLay[1]);
	__syncthreads();

	if(threadIdx.x < 32)
		f2[threadIdx.x] = FourToOneTop(a64, f1, kLay[2]);
	__syncthreads();

	if(threadIdx.x < 128)
		f3[threadIdx.x] = FourToOneTop(a256, f2, kLay[3]);
	__syncthreads();	// threadIdx.x < 512

	out[threadIdx.x] = FourToOneTop(src, f3, kLay[4]);// __half2half2(fup) + kf * kLay[4];
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

	if(threadIdx.x < 8)
		f1[threadIdx.x] = FourToOneTop(a16, f0, kLay[1]);
	__syncthreads();

	out[threadIdx.x] = FourToOneTop(src, f1, kLay[2]);
}// ----------------------------------------------------------------------------------------------

// 128 thread: in_val[4](4*64=256bit) out[128](half2)
static __global__ void glTop2(const uint64_t* __restrict__ in_val, __half2* __restrict__ out){
	__shared__ uint64_t src[4], a4, a16, a64;
	__shared__ __half2 f0[2], f1[8], f2[32];

	if(threadIdx.x < 4)
		src[threadIdx.x] = in_val[threadIdx.x];
	__syncwarp();

	if(threadIdx.x == 0){
		a64 = reduct64by1bit(src);
		get_topA4A16(a64, a4, a16); // a4: 4 bit used only; a16: 16 bit used only;
	}
	__syncthreads();

	if(threadIdx.x < 2){
		const __half* pkf_1 = &kF4[a4 * 4];
		const __half2* pkf2_1 = reinterpret_cast<const __half2*>(pkf_1);
		f0[threadIdx.x] = pkf2_1[threadIdx.x] * kLay[0];
	}
	__syncthreads();

	if(threadIdx.x < 8)
		f1[threadIdx.x] = FourToOneTop(a16, f0, kLay[1]);
	__syncthreads();

	if(threadIdx.x < 32)
		f2[threadIdx.x] = FourToOneTop(a64, f1, kLay[2]);
	__syncthreads();	// threadIdx.x < 128

	out[threadIdx.x] = FourToOneTop(src, f2, kLay[3]);
}// ----------------------------------------------------------------------------------------------

// 512 thread: in_val[16](16*64=1024bit) out[512](half2)
static __global__ void glTop3(const uint64_t* __restrict__ in_val, __half2* __restrict__ out){
	__shared__ uint64_t src[16], a4, a16, a64, a256[4];
	__shared__ __half2 f0[2], f1[8], f2[32], f3[128];

	if(threadIdx.x < 16)
		src[threadIdx.x] = in_val[threadIdx.x];
	__syncwarp();

	if(threadIdx.x < 4)
		a256[threadIdx.x] = reduct64by1bit(src + threadIdx.x * 4);
	__syncwarp();

	if(threadIdx.x == 0){
		a64 = reduct64by1bit(a256);
		get_topA4A16(a64, a4, a16); // a4: 4 bit used only; a16: 16 bit used only;
	}
	__syncthreads();

	if(threadIdx.x < 2){
		const __half* pkf_1 = &kF4[a4 * 4];
		const __half2* pkf2_1 = reinterpret_cast<const __half2*>(pkf_1);
		f0[threadIdx.x] = pkf2_1[threadIdx.x] * kLay[0];
	}
	__syncthreads();

	if(threadIdx.x < 8)
		f1[threadIdx.x] = FourToOneTop(a16, f0, kLay[1]);
	__syncthreads();

	if(threadIdx.x < 32)
		f2[threadIdx.x] = FourToOneTop(a64, f1, kLay[2]);
	__syncthreads();

	if(threadIdx.x < 128)
		f3[threadIdx.x] = FourToOneTop(a256, f2, kLay[3]);
	__syncthreads();	// threadIdx.x < 512

	out[threadIdx.x] = FourToOneTop(src, f3, kLay[4]);// __half2half2(fup) + kf * kLay[4];
}// ----------------------------------------------------------------------------------------------
