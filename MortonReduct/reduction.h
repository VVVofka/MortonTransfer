// reduction.h
#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

	//__device__ __host__ __forceinline__ uint64_t reduct64by1bit(const uint64_t* __restrict__ src);
	__device__ void tstfunc3();

#ifdef __cplusplus
}
#endif