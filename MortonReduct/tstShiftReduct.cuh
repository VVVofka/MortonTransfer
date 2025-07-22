#pragma once
// tstShiftReduct.cuh
#include "common.cuh"
#include "CudaArray.h"
#include "constmem.cuh"
#include <vector>
#include "morton.cuh"

namespace TST_ShiftReduce{
std::vector<uint64_t> fillrnd_1bit(unsigned sz_side){
	std::vector<uint64_t> ret(sz_side * sz_side / 64);
	for(size_t nword = 0; nword < ret.size(); nword++){
		ret[nword] = 0;
		for(uint64_t j = 0; j < 64; j++){
			if((rand() & 0xFF) > 170)
				ret[nword] |= 1ull << j;
		}
	}
	return ret;
}
void dump_cudaar(CudaArray<uint64_t>& cudaar, const std::string& caption = ""){
	printf("%s\n", caption.c_str());
	std::vector<uint64_t> v = cudaar.get_vector();
	for(unsigned y = 0; y < cudaar.szside * 8; y++){
		if((y % 4) == 0 && y){
			if((y % 16) == 0)
				for(unsigned z = 0; z < cudaar.szside * 10 - 1; z++)
					printf("=");
			printf("\n");
		}
		for(unsigned x = 0; x < cudaar.szside * 8; x++){
			unsigned idmorton = EncodeMorton2(x, y);
			//printf("y:%u x:%u id_mortn=%u\n", y, x, idmorton);
			unsigned val = (v[idmorton / 64] >> (idmorton % 64)) & 1;
			if((x % 4) == 0 && x){
				if((x % 16) == 0)
					printf("|");
				else
					printf(" ");
			}
			if(val)
				printf("H");
			else
				printf(".");
		}
		printf("\n");
	}
}

int test01(){
	std::vector<uint64_t> vin = fillrnd_1bit(32);
	CudaArray<uint64_t> lay_in(vin);
	CudaArray<uint64_t> lay_shift(4);
	CudaArray<uint64_t> lay_mid(2);
	CudaArray<uint64_t> lay_top(1);

	setSZ0toConstantMem(32);

	int2 shift{-8,-4};
	dump_cudaar(lay_in, "IN");

	//std::string capt_shift0 = "SHIFT before " + std::to_string(shift.x) + "*" + std::to_string(shift.y);
	//dump_cudaar(lay_shift, capt_shift);
	
	glShiftReduction62by1X4_mid << <1, 1 >> > (lay_in.pdevice, lay_shift.pdevice, lay_mid.pdevice, lay_top.pdevice, shift);
	CHECK_CUDA(cudaDeviceSynchronize());

	std::string capt_shift1 = "\nSHIFT after " + std::to_string(shift.x) + "*" + std::to_string(shift.y);
	dump_cudaar(lay_shift, capt_shift1);

	dump_cudaar(lay_mid, "MID");
	dump_cudaar(lay_top, "TOP");
	return 0;
}
}