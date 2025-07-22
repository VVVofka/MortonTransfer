#pragma once
// tstShiftReduct.cuh
#include "common.cuh"
#include "CudaArray.h"
#include "constmem.cuh"
namespace TST_ShiftReduce{
void fillrnd_1bit(uint64_t* p, size_t size){
	for(size_t i = 0; i < size; i++){
		p[i] = 0;
		for(uint64_t j = 0; j < 64; j++){
			if((rand() & 0xFF) > 170)
				p[i] |= 1ull << j;
		}
	}
}

int test01(){
	CudaArray<uint64_t> lay_in(4);
	CudaArray<uint64_t> lay_shift(4);
	CudaArray<uint64_t> lay_mid(2);
	CudaArray<uint64_t> lay_top(1);
	fillrnd_1bit(lay_in.phost, lay_in.szall);

	setSZ0toConstantMem(32);
	return 0;
}
}