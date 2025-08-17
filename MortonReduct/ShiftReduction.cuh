#pragma once
// ShiftReduction.cuh
#include "shift.cuh"
#include "reduce.cuh"

static __global__ void glShiftReduction62by1X4(const uint64_t* __restrict__ data_in,
					   uint64_t* __restrict__ data_shift,
					   uint64_t* __restrict__ data_top,
					   const int2 shift){
	uint64_t data_mid[4];
	const unsigned up_id_word = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned base_mid_id_word = up_id_word * 4;
#pragma unroll
	for(unsigned i = 0; i < 4; i++){	// by dn word
		const unsigned cur_mid_id_word = base_mid_id_word + i;
		uint64_t data_bottom[4];	// TODO: use data_shift?
#pragma unroll
		for(unsigned j = 0; j < 4; j++){
			const unsigned id = cur_mid_id_word * 4 + j;
			data_shift[id] = data_bottom[j] = shift64by1bit(data_in, id * 64, shift);
		}
		data_mid[i] = reduct64by1bit(data_bottom);
	}
	data_top[up_id_word] = reduct64by1bit(data_mid);
}// -----------------------------------------------------------------------------------------------
static __global__ void glShiftReduction62by1X4_mid(const uint64_t* __restrict__ data_in,
					   uint64_t* __restrict__ data_shift,
					   uint64_t* __restrict__ data_mid,
					   uint64_t* __restrict__ data_top,
					   const int2 shift){
	const unsigned up_id_word = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned base_mid_id_word = up_id_word * 4;
#pragma unroll
	for(unsigned i = 0; i < 4; i++){	// by dn word
		const unsigned cur_mid_id_word = base_mid_id_word + i;
		uint64_t data_bottom[4];	// TODO: use data_shift?
#pragma unroll
		for(unsigned j = 0; j < 4; j++){
			const unsigned id = cur_mid_id_word * 4 + j;
			data_shift[id] = data_bottom[j] = shift64by1bit(data_in, id * 64, shift);
		}
		data_mid[cur_mid_id_word] = reduct64by1bit(data_bottom);
	}
	data_top[up_id_word] = reduct64by1bit(data_mid + base_mid_id_word);
}// -----------------------------------------------------------------------------------------------
