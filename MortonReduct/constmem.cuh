#pragma once
// constmem.cuh
#include "common.cuh"
#include <cuda_fp16.h>
#include <vector>
#include <cassert>

__constant__ __half kF4[16 * 4];
__constant__ __half2 kLay[16];
__constant__ unsigned SZ0;


static void setSZ0toConstantMem(unsigned sz0){
	CHECK_CUDA(cudaMemcpyToSymbol(static_cast<const void*>(&SZ0), &sz0, sizeof(sz0), 0, cudaMemcpyHostToDevice));
}

template <typename T>
static void loadKLay(const T* v, size_t cnt){
	assert(cnt <= 16);
	__half2 vh2[16];
	for(size_t j = 0; j < cnt; j++)
		vh2[j] = j < 16 ? __half2(__half(v[j]), __half(v[j])) : __half2(0, 0);
	assert(sizeof(kLay) == sizeof(vh2));
	CHECK_CUDA(cudaMemcpyToSymbol(kLay, vh2, sizeof(vh2), 0, cudaMemcpyHostToDevice));
}
template <typename T>
static void loadKLay(const std::vector<T>& v){ loadKLay(v.data(), v.size());}


template <typename T>
static void loadKF4(const T* v, size_t cnt){
	assert(cnt <= 16 * 4);
	__half vh[16 * 4];
	for(size_t j = 0; j < cnt; j++)
		vh[j] = (j < 16 * 4) ? __half(v[j]) : __half(0);
	assert(sizeof(kF4) == sizeof(vh));
	CHECK_CUDA(cudaMemcpyToSymbol(kF4, vh, sizeof(vh), 0, cudaMemcpyHostToDevice));
}
template <typename T>
static void loadKF4(const std::vector<T>& v){ loadKF4(v.data(), v.size()); }


static void initConstantMemory(){
	loadKF4<double>({
		-1.0, -1.0, -1.0, -1.0,	//0000
		+0.0, +1.0, +1.0, +1.0,	//0001
		+1.0, +0.0, +1.0, +1.0,	//0010
		+0.0, +0.0, +0.0, +0.0,	//0011
		+1.0, +1.0, +0.0, +1.0,	//0100
		+0.0, +0.0, +0.0, +0.0,	//0101
		+0.0, +0.0, +0.0, +0.0,	//0110
		+0.0, -1.0, -1.0, -1.0,	//0111
		+1.0, +1.0, +1.0, +0.0,	//1000
		+0.0, +0.0, +0.0, +0.0,	//1001
		+0.0, +0.0, +0.0, +0.0,	//1010
		-1.0, -1.0, +0.0, -1.0,	//1011
		+0.0, +0.0, +0.0, +0.0,	//1100
		-1.0, +0.0, -1.0, -1.0,	//1101
		+0.0, -1.0, -1.0, -1.0,	//1110
		-1.0, -1.0, -1.0, -1.0	//1111
	});
	loadKLay<double>({
		0.1, 0.2, 0.3, 0.4, 
		0.5, 0.7, 0.9, 1.0, 
		1.1, 1.2, 1.4, 1.5, 
		2.0, 2.2, 2.4, 3.0
	});
}
