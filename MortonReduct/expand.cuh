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
// blockDim.x = 16 (by 2xhalf2)
// Per thread: half f_up; 1bit mid0; 2bit mid1; 8 bit in;
// Per block: half2 f_up; 32bit mid0; 8bit mid1; 4*32bit in;
static __device__ __forceinline__
void x64(const __half2* __restrict__ pfup,
			const uint32_t* __restrict__ pmida0,
			const uint32_t* __restrict__ pmida1,
			const uint32_t* __restrict__ pdna,
			const __half2* __restrict__ klays,
			__half2* __restrict__ pdst){
	__shared__ __half2 fup2;
	__shared__ __half2 midf1[4];
	__shared__ uint32_t mida1;	// 8 bit only used
	__shared__ uint32_t mida0;	// 32 bit used

	if(threadIdx.x == 0){
		fup2 = pfup[blockIdx.x];
		mida1 = pmida1[blockIdx.x / 4];
		mida0 = pmida0[blockIdx.x];
	}
	syncwarp();
	if(threadIdx.x < 4){
		const uint32_t maska = (mida1 << (threadIdx.x * 8)) & 0xFF;
		const __half2 kf = reinterpret_cast<const __half2*>(kF4 + maska * 4)[threadIdx.x & 1];
		const __half fup = threadIdx.x & 2 ? fup2.y : fup2.x;
		midf1[threadIdx.x] = kf * klays[2] + __half2half2(fup);
	}
	syncwarp();

	const unsigned id_in = blockIdx.x * 8 + threadIdx.x;
	uint32_t tmp = uint32_t(pack16(convolution64to16(psrc[id_in])));
	uint32_t& m = mid[threadIdx.x / 2];
	m = threadIdx.x & 1 ? 0 : tmp;
	syncwarp();
	if(threadIdx.x & 1)
		m |= tmp << 16;
	syncwarp();

	if((threadIdx.x & 1) == 0)
		m = pack8(convolution32to8(m));
	syncwarp();


	const unsigned id_out = blockIdx.x * 8 + threadIdx.x;
	uint32_t tmp = uint32_t(pack16(convolution64to16(psrc[id_in])));
	uint32_t& m = mid[threadIdx.x / 2];
	m = threadIdx.x & 1 ? 0 : tmp;
	syncwarp();
	if(threadIdx.x & 1)
		m |= tmp << 16;
	syncwarp();

	if((threadIdx.x & 1) == 0)
		m = pack8(convolution32to8(m));
	syncwarp();

	if(threadIdx.x == 0)
		*pdst = mid[0] | (mid[1] << 8) | (mid[2] << 16) | (mid[3] << 24);
} // ----------------------------------------------------------------------------------------------
} // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@