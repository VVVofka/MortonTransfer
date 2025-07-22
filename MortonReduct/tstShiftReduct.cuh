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

	uint64_t* pin = lay_in.pdevice;
	uint64_t* pshift = lay_shift.pdevice;
	uint64_t* pmid = lay_mid.pdevice;
	uint64_t* ptop = lay_top.pdevice;
	int2 shift{0,0};
	lay_in.print("IN");
	glShiftReduction62by1X4_mid<<<1,1>>>(pin, pshift, pmid, ptop, shift);
	lay_shift.print("SHIFT");
	lay_mid.print("MID");
	lay_top.print("TOP");

	return 0;
}
}