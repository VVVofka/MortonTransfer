#pragma once
// expand.cuh
#include <cuda_runtime.h>
#include "common.cuh"
#include "pack.cuh"
#include "convolution.cuh"
#include "constmem.cuh"

 // ===============================================================================================
__device__ __forceinline__ __half2 FourToOneTop(const uint64_t maska,
		const __half2* __restrict__ f_prev,
		const __half2 klay){
	const __half2 kf = reinterpret_cast<const __half2*>(kF4 + maska * 4)[threadIdx.x & 1];
	const __half fup = (reinterpret_cast<const __half*>(f_prev))[threadIdx.x >> 1];
	return  __half2half2(fup) + kf * klay;
} // ===============================================================================================

namespace Expand{
// blockDim.x = 64 (by 2xhalf2)
static __device__ __inline__
void x64from32_64thr(const __half2* __restrict__ pfup2,	//	[4]
			const uint32_t* __restrict__ pmida,	// [1+4]
			const uint32_t* __restrict__ pdna,	// [16]
			const __half2* __restrict__ klays,	// [3]
			__half2* __restrict__ pdst){		// [256]

	__shared__ __half2 midf1[16];
	const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;

	const uint32_t& mida1 = pmida[blockIdx.x * 5];
	const uint32_t* mida0 = &mida1 + 1;	// pmida[1] ... pmida[4]

	if(threadIdx.x < 16){
		const __half* pfup = reinterpret_cast<const __half*>(pfup2);	// [8] <- [4]
		const __half2 fup = __half2half2(pfup[id / 8]);
		const uint32_t shift = (threadIdx.x & 0b1110) * 2;
		const uint32_t maska = (mida1 >> shift) & 0b1111;
		const __half2 kf = reinterpret_cast<const __half2*>(kF4 + maska * 4)[threadIdx.x & 1];
		midf1[threadIdx.x] = kf * klays[2] + fup;
	}
	syncwarp();
	// threadIdx.x < 64
	const __half* pfup = reinterpret_cast<const __half*>(midf1);	// [32] <- [16]
	__half2 fup = __half2half2(pfup[threadIdx.x / 2]);
	const uint32_t& wordmaska = mida0[threadIdx.x / 16];
	const uint32_t shift = threadIdx.x & 0b1100;
	const uint32_t maska = (wordmaska >> shift) & 0b1111;
	const __half2 kf = reinterpret_cast<const __half2*>(kF4 + maska * 4)[threadIdx.x & 1];
	fup += kf * klays[1];	// 64

	__half2* dst = pdst + id * 4;
	const uint32_t dna = pdna[blockIdx.x * 16 + threadIdx.x / 4];
	const uint32_t wordmaskadn = dna >> (((threadIdx.x >> 1) & 1) * 16);

#pragma unroll
	for(unsigned n = 0; n < 2; n++){	// 128
		__half2 fup_in = __half2half2(reinterpret_cast<const __half*>(&fup)[n]);
#pragma unroll
		for(unsigned j = 0; j < 2; j++){
			const uint32_t i = n * 2 + j;	// 0...3
			const uint32_t maska = (wordmaskadn >> (i * 4)) & 0b1111;
			const __half2 kf = reinterpret_cast<const __half2*>(kF4 + maska * 4)[j];
			dst[i] = kf * klays[0] + fup_in;
		}
	}
} // ----------------------------------------------------------------------------------------------
// blockDim.x = 256 (by 2xhalf2)
static __device__ __inline__
void x64from32_256thr(const __half2* __restrict__ pfup2,	// [4]
			const uint32_t* __restrict__ pmida,				// [1+4]
			const uint32_t* __restrict__ pdna,				// [16]
			const __half2* __restrict__ klays,				// [3]
			__half2* __restrict__ pdst){					// [256]

	__shared__ __half2 midf1[16];
	__shared__ __half2 midf0[64];
	const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;

	const uint32_t& mida1 = pmida[blockIdx.x * 5];
	const uint32_t* mida0 = &mida1 + 1;	// pmida[1] ... pmida[4]

	if(threadIdx.x < 16){
		const __half* pfup = reinterpret_cast<const __half*>(pfup2);	// [8] <- [4]
		const __half2 fup = __half2half2(pfup[id / 8]);
		const uint32_t shift = (threadIdx.x & 0b1110) * 2;
		const uint32_t maska = (mida1 >> shift) & 0b1111;
		const __half2 kf = reinterpret_cast<const __half2*>(kF4 + maska * 4)[threadIdx.x & 1];
		midf1[threadIdx.x] = kf * klays[2] + fup;
	}
	syncwarp();

	if(threadIdx.x < 64){
		const __half* pfup = reinterpret_cast<const __half*>(midf1);	// [32] <- [16]
		const __half2 fup = __half2half2(pfup[threadIdx.x / 2]);
		const uint32_t& wordmaska = mida0[threadIdx.x / 16];
		const uint32_t shift = threadIdx.x & 0b1100;
		const uint32_t maska = (wordmaska >> shift) & 0b1111;
		const __half2 kf = reinterpret_cast<const __half2*>(kF4 + maska * 4)[threadIdx.x & 1];
		midf0[threadIdx.x] = kf * klays[1] + fup;
	}
	syncthreads();

	// threadIdx.x < 256
	const uint32_t dna = pdna[blockIdx.x * 16 + threadIdx.x / 16];
	const uint32_t wordmaskadn = (dna >> ((threadIdx.x & 0b111) * 4)) & 0b1111;
	const __half2 kf = reinterpret_cast<const __half2*>(kF4 + wordmaskadn * 4)[threadIdx.x & 1];
	const __half* pfup = reinterpret_cast<const __half*>(midf0);	// [128] <- [64]
	const __half2 fup = __half2half2(pfup[threadIdx.x / 2]);
	pdst[id] = kf * klays[0] + fup;
} // ----------------------------------------------------------------------------------------------
} // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@