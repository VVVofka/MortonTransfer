#pragma once
// top.cuh
#include "reduction.cuh"
#include "constmem.cuh"
#include <cuda_fp16.h>
#include <cuda/std/cassert>
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
static __device__ __host__ __inline__ uint64_t get_topA4A16(const uint64_t src){
	// 64 -> 16 unpack
	constexpr uint64_t M = 0x1111'1111'1111'1111ULL;
	uint64_t sum0 = (src & M) + ((src >> 1) & M) + ((src >> 2) & M) + ((src >> 3) & M);
	sum0 >>= 1;			// 1 if 2 or 3
	sum0 |= sum0 >> 1;  // or if 4 ( res in pos 0)
	uint64_t ret = pack16(sum0);

	// 16 unpack -> 4 unpack
	constexpr uint64_t M0 = 0x0001'0001'0001'0001ULL, M1 = M0 << 4, M2 = M1 << 4, M3 = M2 << 4;
	uint64_t sum1 = (sum0 & M0) + ((sum0 & M1) >> 4) + ((sum0 & M2) >> 8) + ((sum0 & M3) >> 12);
	sum1 >>= 1;			// 1 if 2 or 3
	sum1 |= sum1 >> 1;  // or if 4 ( res in pos 0)
	return (ret << 4) | pack4(sum1);
} // ----------------------------------------------------------------------------------------------
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
	__shared__ uint64_t src, a4, a16;
	__shared__ __half2 f0[2], f1[8];

	if(threadIdx.x == 0){
		src = in_val[0];
		get_topA4A16(src, a4, a16); // a4: 4 bit used only; a16: 16 bit used only;
	}
	syncthreads();

	if(threadIdx.x < 2){
		const __half* pkf_1 = &kF4[a4 * 4];
		const __half2* pkf2_1 = reinterpret_cast<const __half2*>(pkf_1);
		f0[threadIdx.x] = pkf2_1[threadIdx.x] * kLay[0];
	}
	syncthreads();

	if(threadIdx.x < 8)
		f1[threadIdx.x] = FourToOneTop(a16, f0, kLay[1]);
	syncthreads();

	out[threadIdx.x] = FourToOneTop(src, f1, kLay[2]);
}// ----------------------------------------------------------------------------------------------

// 128 thread: in_val[4](4*64=256bit) out[128](half2)
static __global__ void glTop2(const uint64_t* __restrict__ in_val, __half2* __restrict__ out){
	__shared__ uint64_t src[4], a4, a16, a64;
	__shared__ __half2 f0[2], f1[8], f2[32];

	if(threadIdx.x < 4)
		src[threadIdx.x] = in_val[threadIdx.x];
	syncwarp();

	if(threadIdx.x == 0){
		a64 = reduct64by1bit(src);
		get_topA4A16(a64, a4, a16); // a4: 4 bit used only; a16: 16 bit used only;
	}
	syncthreads();

	if(threadIdx.x < 2){
		const __half* pkf_1 = &kF4[a4 * 4];
		const __half2* pkf2_1 = reinterpret_cast<const __half2*>(pkf_1);
		f0[threadIdx.x] = pkf2_1[threadIdx.x] * kLay[0];
	}
	syncthreads();

	if(threadIdx.x < 8)
		f1[threadIdx.x] = FourToOneTop(a16, f0, kLay[1]);
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

// 1024 thread: 4 out(2*half2)/thread; in_val[64](64*64=4096values) out[2048](half2)
static __global__ void glTop4(const uint64_t* __restrict__ in_val, __half2* __restrict__ out){
	__shared__ uint64_t src[64], a4_16, a64, a256[4], a1024[16];
	__shared__ uint64_t src32[64];
	__shared__ __half2 f0[2], f1[8], f2[32], f3[128], f4[512];

	if(threadIdx.x < 64){
		src[threadIdx.x] = in_val[threadIdx.x];
		const auto shift = (threadIdx.x / 16) * 16;
		src32[threadIdx.x] = reduct64to16bit(src[threadIdx.x]) << shift;
	}
	syncthreads();

	if(threadIdx.x < 16){
		uint64_t* p = src32 + threadIdx.x * 4;
		a1024[threadIdx.x] = p[0] | p[1] | p[2] | p[3];
	}
	syncwarp();

	if(threadIdx.x < 4)
		a256[threadIdx.x] = reduct64by1bit(a1024 + threadIdx.x * 4);
	syncwarp();

	if(threadIdx.x == 0){
		a64 = reduct64by1bit(a256);
		a4_16 = get_topA4A16(a64); // a4_16: 0-3 bit for a4; 4-19 bit for a16
	}
	syncwarp();

	if(threadIdx.x < 2){
		const __half* pkf_1 = &kF4[(a4_16 & 15) * 4];
		const __half2* pkf2_1 = reinterpret_cast<const __half2*>(pkf_1);
		f0[threadIdx.x] = pkf2_1[threadIdx.x] * kLay[0];
	}
	syncwarp();

	if(threadIdx.x < 8)
		f1[threadIdx.x] = FourToOneTop((a4_16 >> 4), f0, kLay[1]);
	syncwarp();

	if(threadIdx.x < 32)
		f2[threadIdx.x] = FourToOneTop(a64, f1, kLay[2]);
	syncthreads();

	if(threadIdx.x < 128)
		f3[threadIdx.x] = FourToOneTop(a256, f2, kLay[3]);
	syncthreads();

	if(threadIdx.x < 512)
		f4[threadIdx.x] = FourToOneTop(a1024, f3, kLay[4]);// __half2half2(fup) + kf * kLay[4];
	syncthreads();
	// threadIdx.x < 1024
	const auto shift = 4 * (threadIdx.x & 15);
	const uint32_t maska = uint32_t(src[threadIdx.x / 16] >> shift) & 0xF;
	const __half2* kf = reinterpret_cast<const __half2*>(kF4 + maska * 4);
	const __half2 fup = __half2half2((reinterpret_cast<const __half*>(f4))[threadIdx.x / 2]);
	// threadIdx[1024] out[2048]
	__half2* pout = out + threadIdx.x * 2;
	pout[0] = fup + kf[0] * kLay[5];
	pout[1] = fup + kf[1] * kLay[5];
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
// GridDim.x = 1024 blocks; BlockDim.x = 64 threads;
// Parametrs:
// uint64_t data_in[65536]; 4'194'304 values (2048x2048)
// uint32_t data_out[1024] // use 64 bit
#define DUMPSRC(A, C) dumpsrc(A, sizeof(A)/sizeof(A[0]), C)

static __global__ void glUpMid3(const uint64_t* __restrict__ data_in,
								uint32_t* __restrict__ data_out){
	__shared__ uint64_t shr[64];
	const auto id_in = blockIdx.x * 64 + threadIdx.x;
	shr[threadIdx.x] = reduct64to1(data_in[id_in]) << threadIdx.x;
	syncthreads();
	if(threadIdx.x == 0){
		uint64_t ret = shr[0];
#pragma unroll
		for(unsigned j = 1; j < 64; j++)
			ret |= shr[j];
		data_out[blockIdx.x] = ret;	// 1024
	}
}// ===============================================================================================
// GridDim.x = 1024 blocks; BlockDim.x = 1024 threads; idThread[1048576] 4 value/thread
// Parametrs:
// __half data_fup[1024] 
// uint64_t data_in[65536]; 4'194'304 values (2048x2048)
// __half data_out[65536];  4'194'304 values (2048x2048)
#define DUMPSRC(A, C) dumpsrc(A, sizeof(A)/sizeof(A[0]), C)

static __global__ void glDnMid3(const uint64_t* __restrict__ data_in,
								uint32_t* __restrict__ data_out){
	__shared__ uint64_t shr[64];
	const auto id_in = blockIdx.x * 1024 + threadIdx.x;
	const uint64_t datain = data_in[id_in / 16];

	shr[threadIdx.x] = reduct64to1() << threadIdx.x;
	syncthreads();
	if(threadIdx.x == 0){
		uint64_t ret = shr[0];
#pragma unroll
		for(unsigned j = 1; j < 64; j++)
			ret |= shr[j];
		data_out[blockIdx.x] = ret;	// 1024
	}
}// ===============================================================================================

