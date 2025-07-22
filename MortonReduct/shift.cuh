#pragma once
//shift.cuh
#include "common.cuh"
#include "morton.cuh"
#include "constmem.cuh"

#include <vector>
static __device__ __inline__ uint64_t shift64by1bit(const uint64_t* __restrict__ data_in, 
						unsigned id_morton, // blockIdx.x * blockDim.x + threadIdx.x
						const int2& shift){
	uint64_t ret = 0;
#pragma unroll
	for(uint64_t i = 0; i < 64ull; ++i){	// bits in word
		const unsigned shift_decart_x = DecodeMorton2X(id_morton);
		const unsigned shift_decart_y = DecodeMorton2Y(id_morton);
#ifdef _DEBUG
		if(shift_decart_x >= SZ0 || shift_decart_y >= SZ0){ printf("out of range!\n"); return 0; }
#endif // _DEBUG
		const unsigned in_decart_x = (shift_decart_x + SZ0 - shift.x) & (SZ0 - 1);
		const unsigned in_decart_y = (shift_decart_y + SZ0 - shift.y) & (SZ0 - 1);

		const unsigned in_morton_id = EncodeMorton2(in_decart_x, in_decart_y);
		const unsigned id_word_in = in_morton_id / 64;
		const uint64_t word = data_in[id_word_in];
		const uint64_t shift_word = in_morton_id & 63;
		const uint64_t val = (word >> shift_word) & 1ull;
		
		ret = (ret & ~(1ull << i)) | (val << i);
		id_morton++;
	}
	return ret;
}// ************************************************************************************