#pragma once
//shift.cuh
#include "common.cuh"
#include <vector>
extern __constant__ unsigned SZ0;
__device__ void __inline__ shift64by1bit(const uint64_t* __restrict__ data_in,
					   uint64_t* __restrict__ data_shift,
					   const int2& shift){
	const unsigned shift_id_word = blockIdx.x * blockDim.x + threadIdx.x; // data_shift index
	uint64_t new_shift_word = data_shift[shift_id_word];
#pragma unroll
	for(unsigned i = 0; i < 64; ++i){	// bits in word
		const unsigned shift_morton_id = shift_id_word * 64 + i;
		const unsigned shift_decart_x = DecodeMorton2X(shift_morton_id);
		const unsigned shift_decart_y = DecodeMorton2Y(shift_morton_id);
#ifdef _DEBUG
		if(shift_decart_x >= SZ0 || shift_decart_y >= SZ0){ printf("out of range!\n"); return; }
#endif // _DEBUG
		const unsigned in_decart_x = (shift_decart_x + SZ0 - shift.x) & (SZ0 - 1);
		const unsigned in_decart_y = (shift_decart_y + SZ0 - shift.y) & (SZ0 - 1);

		const unsigned in_morton_id = EncodeMorton2(in_decart_x, in_decart_y);
		const unsigned val = (data_in[in_morton_id / 64] >> i) & 1;
		new_shift_word = (new_shift_word & ~(1 << i)) | (val << i);
	}
	data_shift[shift_id_word] = new_shift_word;
}// ************************************************************************************