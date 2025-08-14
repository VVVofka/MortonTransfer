#pragma once
// PackConvolution.cuh
#include "convolution.cuh"
#include "pack.cuh"

namespace PackConvolution{
// psrc[2]
static __device__ __host__ __forceinline__
uint32_t from128to32(const uint64_t* __restrict__ psrc){
	
	const uint64_t conv0 = Convolution::from64to16(psrc[0]);
	const uint32_t pack0 = uint32_t(pack16(conv0));

	const uint64_t conv1 = Convolution::from64to16(psrc[1]);
	const uint32_t pack1 = uint32_t(pack16(conv1));
	
	return pack0 | (pack1 << 16);
} // ==============================================================================================
// psrc[4]
static __device__ __host__ __forceinline__
uint32_t from128to32(const uint32_t* __restrict__ psrc32){
	return from128to32(reinterpret_cast<const uint64_t*>(psrc32));
} // ==============================================================================================
static __device__ __host__ __forceinline__
uint32_t from64to16(const uint64_t psrc){
	const uint64_t conv = Convolution::from64to16(psrc);
	const auto pack = pack16(conv);
	return uint32_t(pack);
} // ==============================================================================================
static __device__ __host__ __forceinline__
uint32_t from32to8(const uint32_t psrc){
	const uint32_t conv = Convolution::from32to8(psrc);
	const uint32_t pack = pack8(conv);
	return pack;
} // ==============================================================================================

} // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
