#pragma once
// expand.cuh
#include <cuda_runtime.h>
#include "common.cuh"
#include "pack.cuh"
#include "convolution.cuh"
#include "constmem.cuh"

 // ===============================================================================================
__device__ __forceinline__ __half2 FourToOneTop(const uint64_t maska, const __half2* f_prev, const __half2 klay){
	const __half2 kf = reinterpret_cast<const __half2*>(kF4 + maska * 4)[threadIdx.x & 1];
	const __half fup = (reinterpret_cast<const __half*>(f_prev))[threadIdx.x >> 1];
	return  __half2half2(fup) + kf * klay;
} // ===============================================================================================


// half2 f_up[16] *2=32
// blockDim.x = 128 (by 2xhalf2)
// pdst[8]
static __device__ __forceinline__
void expand16for64bit_maxThread(const __half2* __restrict__ f_up, 
								const uint64_t* __restrict__ a_dn, 
								__half2* __restrict__ pdst){
	__shared__ __half2 midf[64];
	__shared__ uint32_t mida[4];
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
