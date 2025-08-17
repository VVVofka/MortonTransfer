#pragma once
// top.cuh
#include "reduce.cuh"
#include "constmem.cuh"
#include <cuda_fp16.h>
#include <cuda/std/cassert>
// ----------------------------------------------------------------------------------------------
__device__ __forceinline__ __half2 FourToOneTop(const uint64_t* src, const __half2* f_prev, const __half2 klay){
	const uint64_t maska = (src[threadIdx.x / 32] >> (4 * ((threadIdx.x & 31) / 2))) & 0xF;
	const __half2 kf = reinterpret_cast<const __half2*>(kF4 + maska * 4)[threadIdx.x & 1];
	const __half fup = (reinterpret_cast<const __half*>(f_prev))[threadIdx.x / 2];
	return __half2half2(fup) + kf * klay;
} // ----------------------------------------------------------------------------------------------
__device__ __forceinline__ __half2 FourToOneTop(const uint64_t src, const __half2* f_prev, const __half2 klay){
	const uint64_t maska = (src >> (4 * (threadIdx.x / 2))) & 0xF;	// wo (threadIdx.x & 31)
	const __half2 kf = reinterpret_cast<const __half2*>(kF4 + maska * 4)[threadIdx.x & 1];
	const __half fup = (reinterpret_cast<const __half*>(f_prev))[threadIdx.x / 2];
	return  __half2half2(fup) + kf * klay;
} // ===============================================================================================

// 32 thread: in_val[1](64bit) out[32](half2)
static __global__ void glTop1(const uint64_t* __restrict__ in_val, __half2* __restrict__ out){
	__shared__ uint64_t src, a4_16;
	__shared__ __half2 f0[2], f1[8];

	if(threadIdx.x == 0){
		src = in_val[0];
		a4_16 = get_topA4A16(src); // a4: 4 bit used only; a16: 16 bit used only;
	}
	syncthreads();

	if(threadIdx.x < 2){
		const __half* pkf_1 = &kF4[(a4_16 & 15) * 4];
		const __half2* pkf2_1 = reinterpret_cast<const __half2*>(pkf_1);
		f0[threadIdx.x] = pkf2_1[threadIdx.x] * kLay[0];
	}
	syncthreads();

	if(threadIdx.x < 8)
		f1[threadIdx.x] = FourToOneTop((a4_16 >> 4), f0, kLay[1]);
	syncthreads();

	out[threadIdx.x] = FourToOneTop(src, f1, kLay[2]);
}// ----------------------------------------------------------------------------------------------

// 128 thread: in_val[4](4*64=256bit) out[128](half2)
static __global__ void glTop2(const uint64_t* __restrict__ in_val, __half2* __restrict__ out){
	__shared__ uint64_t src[4], a4_16, a64;
	__shared__ __half2 f0[2], f1[8], f2[32];

	if(threadIdx.x < 4)
		src[threadIdx.x] = in_val[threadIdx.x];
	syncwarp();

	if(threadIdx.x == 0){
		a64 = reduct64by1bit(src);
		a4_16 = get_topA4A16(a64); // a4: 4 bit used only; a16: 16 bit used only;
	}
	syncthreads();

	if(threadIdx.x < 2){
		const __half* pkf_1 = &kF4[(a4_16 & 15) * 4];
		const __half2* pkf2_1 = reinterpret_cast<const __half2*>(pkf_1);
		f0[threadIdx.x] = pkf2_1[threadIdx.x] * kLay[0];
	}
	syncthreads();

	if(threadIdx.x < 8)
		f1[threadIdx.x] = FourToOneTop((a4_16 >> 4), f0, kLay[1]);
	syncthreads();

	if(threadIdx.x < 32)
		f2[threadIdx.x] = FourToOneTop(a64, f1, kLay[2]);
	syncthreads();	// threadIdx.x < 128

	out[threadIdx.x] = FourToOneTop(src, f2, kLay[3]);
}// ----------------------------------------------------------------------------------------------

// 512 thread: in_val[16](16*64=1024bit) out[512](half2)
static __global__ void glTop3(const uint64_t* __restrict__ in_val, __half2* __restrict__ out){
	__shared__ uint64_t src[16], a4_16, a64, a256[4];
	__shared__ __half2 f0[2], f1[8], f2[32], f3[128];

	if(threadIdx.x < 16)
		src[threadIdx.x] = in_val[threadIdx.x];
	syncwarp();

	if(threadIdx.x < 4)
		a256[threadIdx.x] = reduct64by1bit(src + threadIdx.x * 4);
	syncwarp();

	if(threadIdx.x == 0){
		a64 = reduct64by1bit(a256);
		a4_16 = get_topA4A16(a64); // a4: 4 bit used only; a16: 16 bit used only;
	}
	syncthreads();

	if(threadIdx.x < 2){
		const __half* pkf_1 = &kF4[(a4_16 & 15) * 4];
		const __half2* pkf2_1 = reinterpret_cast<const __half2*>(pkf_1);
		f0[threadIdx.x] = pkf2_1[threadIdx.x] * kLay[0];
	}
	syncthreads();

	if(threadIdx.x < 8)
		f1[threadIdx.x] = FourToOneTop((a4_16 >> 4), f0, kLay[1]);
	syncthreads();

	if(threadIdx.x < 32)
		f2[threadIdx.x] = FourToOneTop(a64, f1, kLay[2]);
	syncthreads();

	if(threadIdx.x < 128)
		f3[threadIdx.x] = FourToOneTop(a256, f2, kLay[3]);
	syncthreads();;	// threadIdx.x < 512

	out[threadIdx.x] = FourToOneTop(src, f3, kLay[4]);// __half2half2(fup) + kf * kLay[4];
}// ----------------------------------------------------------------------------------------------
// GridDim.x = 512 blocks; BlockDim.x = 1024 threads;
// Lay4: half2[512] f_up; (1'024 half) side=32
// Lay5: half2[2'048] f_up; (4'096 half) side=64
// Lay6: half2[8'192] f_up; (16'384 half) side=128
// Lay7: half2[32'768] f_up; (65'536 half) side=256
// Lay8: half2[131'072] f_up; (262'144 half) side=512
// Lay9: half2[524'288] f_up; (1'048'576 half) side=1024
// Parametrs:
// half2 f_up_in[512] =BlockDim; 1024 values;
// uint64_t in_val[16384] =id/32 (32 per block) - recalculate all 'a' for avoid global memory access.
// half2 fout[524'288] //see Lay9 (1024 per block; 1 per thread)
static __global__ void glDnMid3_(const __half2* __restrict__ f_up_in,
								const uint64_t* __restrict__ in_val,
								__half2* __restrict__ f_out){
	__shared__ uint32_t src_a[64], src_b[7];
	half2 fout = f_up_in[blockIdx.x];

	const uint32_t* src32 = reinterpret_cast<const uint32_t*>(in_val);
	if((threadIdx.x & 15) == 0)
		src_a[threadIdx.x / 16] = src32[blockIdx.x * 64 + threadIdx.x / 16];
	syncthreads();

	{	// Lay 9	in:(64*32bit)
		const uint32_t mask = (src_a[threadIdx.x / 16] >> (4 * ((threadIdx.x & 15) / 4))) & 0xF;
		const __half2 kf = reinterpret_cast<const __half2*>(kF4 + mask * 4)[threadIdx.x & 1];
		fout += kf * kLay[9];

		if((threadIdx.x & 15) == 0){
			uint32_t& src = src_a[threadIdx.x / 16];
			constexpr uint32_t M = 0x1111'1111;
			src = (src & M) + ((src >> 1) & M) + ((src >> 2) & M) + ((src >> 3) & M);
			src >>= 1;			// 1 if 2 or 3
			src |= src >> 1;  // or if 4 ( res in pos 0)
			src = (src & 0x0001'0001) | ((src & 0x0010'0010) >> 3) | ((src & 0x0100'0100) >> 6) | ((src & 0x1000'1000) >> 9);
		}
		syncthreads();
	}
	{	// Lay 8	in:(64*8bit)
		const uint32_t mask = (src_a[threadIdx.x / 16] >> (16 * (threadIdx.x & 1))) & 0xF;
		const __half2 kf = reinterpret_cast<const __half2*>(kF4 + mask * 4)[threadIdx.x & 1];
		fout += kf * kLay[8];
		if((threadIdx.x & 15) == 0)
			src_a[threadIdx.x / 16] = mask;
		syncthreads();
	}
	{	// Lay 7	in:(63*2bit)
		const uint32_t mask = src_a[(threadIdx.x / 32) * 2] | (src_a[(threadIdx.x / 32) * 2 + 1] << 2);
		const __half2 kf = reinterpret_cast<const __half2*>(kF4 + mask * 4)[threadIdx.x & 1];
		fout += kf * kLay[7];
		if((threadIdx.x & 127) == 0)
			src_b[threadIdx.x / 128] = mask;
		syncthreads();  // TODO: syncwarp()?
	}
	{	// Lay 6	in:(8*4bit)
		const uint32_t* base = src_b + threadIdx.x / 128;
		const uint32_t mask = base[0];
		const __half2 kf = reinterpret_cast<const __half2*>(kF4 + mask * 4)[threadIdx.x & 1];
		fout += kf * kLay[6];
		if((threadIdx.x & 127) == 0)
			src_a[threadIdx.x / 128] = get_a(mask);
		syncwarp();
	}
	{	// Lay 5	in:(8*1bit)
		const uint32_t* base = src_a + threadIdx.x / 128;
		const uint32_t mask = base[0] | (base[1] << 1) | (base[2] << 2) | (base[3] << 3);
		const __half2 kf = reinterpret_cast<const __half2*>(kF4 + mask * 4)[threadIdx.x & 1];
		fout += kf * kLay[5];
	}
	f_out[threadIdx.x] = fout;
} // ===============================================================================================

static __device__ void dumpsrc(const uint32_t* v, size_t cnt, const char* c = nullptr){
	if(threadIdx.x + blockIdx.x == 0){
		if(c) printf("\n%u %s\n", (unsigned)cnt, c);
		for(int j = 0; j < cnt; j++){
			if(j && ((j % 8) == 0)) printf("\n");
			if((j % 4) == 0) printf(" ");
			printf("%8X ", v[j]);
		}
		printf("\n");
	}
} // ===============================================================================================
#define DUMPSRC(A, C) dumpsrc(A, sizeof(A)/sizeof(A[0]), C)

// GridDim.x = 32 blocks; BlockDim.x = 32 threads;
// Parametrs:
// uint64_t data_in[16384]; 1'048'576 values (1024x1024)
// uint32_t data_out[32] // 1024 values
static __global__ void glUpMid3(const uint64_t* __restrict__ data_in,
								uint32_t* __restrict__ data_out){
	__shared__ uint32_t shr[32];
	const unsigned id_in = blockIdx.x * 32 + threadIdx.x;
	shr[threadIdx.x] = reduct64to1(data_in[id_in]) << threadIdx.x;
	syncwarp();
	if((threadIdx.x & 0b1) == 0)
		shr[threadIdx.x * 2] |= shr[threadIdx.x * 2 + 1];
	syncwarp();

	if((threadIdx.x & 0b11) == 0)
		shr[threadIdx.x * 4] |= shr[threadIdx.x * 4 + 2];
	syncwarp();
	
	if((threadIdx.x & 0b111) == 0)
		shr[threadIdx.x * 8] |= shr[threadIdx.x * 8 + 4];
	syncwarp();

	data_out[blockIdx.x] = shr[0] | shr[8] | shr[16] | shr[24];	// 1024
}// ===============================================================================================

