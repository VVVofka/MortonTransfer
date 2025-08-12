#pragma once
// expand.cuh
#include <cuda_runtime.h>
#include "common.cuh"
#include "pack.cuh"
#include "convolution.cuh"

//blockDim.x = 8
// pdst[8]
static __device__ __forceinline__
void expand16for64bit_maxThread(const __half2* __restrict__ f_up, 
		const uint64_t* __restrict__ a_dn, 
		__half2* __restrict__ pdst){
	__shared__ uint32_t mid[4];
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
