#pragma once
#include <cuda_runtime.h>
#include <cuda.h>

static __device__ __host__ __forceinline__ unsigned DecodeMorton2X(unsigned code){
	code &= 0x55555555;
	code = (code ^ (code >> 1)) & 0x33333333;
	code = (code ^ (code >> 2)) & 0x0F0F0F0F;
	code = (code ^ (code >> 4)) & 0x00FF00FF;
	code = (code ^ (code >> 8)) & 0x0000FFFF;
	return code;
}

static __device__ __host__ __forceinline__ unsigned DecodeMorton2Y(unsigned code){
	code >>= 1;
	code &= 0x55555555;
	code = (code ^ (code >> 1)) & 0x33333333;
	code = (code ^ (code >> 2)) & 0x0F0F0F0F;
	code = (code ^ (code >> 4)) & 0x00FF00FF;
	code = (code ^ (code >> 8)) & 0x0000FFFF;
	return code;
}

static __device__ __forceinline__ unsigned EncodeMorton2(unsigned x, unsigned y){
	return __brevll(__brev(x) >> 1 | (__brev(y) >> 1) << 1) >> 32;
}

static __host__  inline unsigned EncodeMorton2h(unsigned x, unsigned y){
	x &= 0x0000ffff;
	y &= 0x0000ffff;
	x = (x | (x << 8)) & 0x00FF00FF;
	y = (y | (y << 8)) & 0x00FF00FF;
	x = (x | (x << 4)) & 0x0F0F0F0F;
	y = (y | (y << 4)) & 0x0F0F0F0F;
	x = (x | (x << 2)) & 0x33333333;
	y = (y | (y << 2)) & 0x33333333;
	x = (x | (x << 1)) & 0x55555555;
	y = (y | (y << 1)) & 0x55555555;
	return x | (y << 1);
}
