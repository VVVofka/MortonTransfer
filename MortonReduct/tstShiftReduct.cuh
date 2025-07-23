#pragma once
// tstShiftReduct.cuh
#include "common.cuh"
#include "CudaArray.h"
#include "constmem.cuh"
#include <vector>
#include "morton.cuh"
#include "top.cuh"

namespace TST_ShiftReduce{
std::vector<uint64_t> fillrnd_1bit(unsigned sz_side){
	std::vector<uint64_t> ret(sz_side * sz_side / 64);
	for(size_t nword = 0; nword < ret.size(); nword++){
		ret[nword] = 0;
		for(uint64_t j = 0; j < 64; j++){
			if((rand() & 0xFF) > 180)
				ret[nword] |= 1ull << j;
		}
	}
	return ret;
}
void dump2D_vhost(std::vector<uint64_t>& v, const std::string& caption = ""){
	printf("%s\n", caption.c_str());
	unsigned cntside = (unsigned)sqrt(double(v.size()));
	for(unsigned yr = 0; yr < cntside * 8; yr++){
		unsigned y = cntside * 8 - yr - 1;
		if((yr % 4) == 0 && yr){
			if((yr % 16) == 0)
				for(unsigned z = 0; z < cntside * 10 - 1; z++)
					printf("=");
			printf("\n");
		}
		for(unsigned x = 0; x < cntside * 8; x++){
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
				printf("1");
			else
				printf(".");
		}
		printf("\n");
	}
}
void dump2D_cudaar(CudaArray<uint64_t>& cudaar, const std::string& caption = ""){
	std::vector<uint64_t> v = cudaar.get_vector();
	dump2D_vhost(v, caption);
}
void dump2D_uns64(uint64_t u, const std::string& caption = ""){
	std::vector<uint64_t> v{u};
	dump2D_vhost(v, caption);
}
template <typename T>
void dump1D_uns64(T u, const std::string& caption = ""){
	printf("%s\n", caption.c_str());
	for(T ir = 0; ir < sizeof(T) * 8; ir++){
		T i = sizeof(T) * 8 - ir - 1;
		if((ir % 4) == 0 && ir){
			if((ir % 16) == 0)
				printf(" | ");
			else
				printf(" ");
		}
		if((u >> i) & 1)
			printf("1");
		else
			printf(".");
	}
	printf("\n");
}

int test01(){
	std::vector<uint64_t> vin = fillrnd_1bit(32);
	CudaArray<uint64_t> lay_in(vin);
	CudaArray<uint64_t> lay_shift(4);
	CudaArray<uint64_t> lay_mid(2);
	CudaArray<uint64_t> lay_top(1);

	setSZ0toConstantMem(32);

	int2 shift{-8,-4};
	dump2D_cudaar(lay_in, "IN");

	//std::string capt_shift0 = "SHIFT before " + std::to_string(shift.x) + "*" + std::to_string(shift.y);
	//dump_cudaar(lay_shift, capt_shift);

	glShiftReduction62by1X4_mid << <1, 1 >> > (lay_in.pdevice, lay_shift.pdevice, lay_mid.pdevice, lay_top.pdevice, shift);
	CHECK_CUDA(cudaDeviceSynchronize());

	std::string capt_shift1 = "\nSHIFT after " + std::to_string(shift.x) + "*" + std::to_string(shift.y);
	dump2D_cudaar(lay_shift, capt_shift1);

	dump2D_cudaar(lay_mid, "MID");
	dump2D_cudaar(lay_top, "TOP");
	return 0;
}

int tst_top(unsigned seed = 0){
	srand(seed ? seed : (unsigned)time(0));

	std::vector<uint64_t> vin = fillrnd_1bit(8);
	vin[0] = 0x0FA0'0000'0007'C030;
	dump2D_vhost(vin, "In");
	dump1D_uns64(vin[0], "In");
	__half2* pout = nullptr;
	uint64_t mid = 0, bot = 0;
	uint64_t top = get_top(vin[0], pout, bot, mid);
	dump2D_uns64(bot, "\nBot");
	dump1D_uns64(bot, "\nBot");

	dump2D_uns64(mid, "\nMid");
	dump1D_uns64(mid, "\nMid");

	dump2D_uns64(top, "\nOut");
	dump1D_uns64(top, "\nOut");
	return 0;
}
}