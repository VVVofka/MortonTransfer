#include "reduction.cuh"

namespace PackReduct{
static __device__ __forceinline__
void from128to32by1(){

} // ----------------------------------------------------------------------------------------------

static __device__ __forceinline__
void packReduct128to32(const uint64_t* __restrict__ psrc, uint32_t* __restrict__ pdst){
	PackReduct::from128to32();
} // ----------------------------------------------------------------------------------------------
} // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
