#pragma once
// tstShiftReduct.cuh
#include "common.cuh"
#include "CudaArray.h"
#include "constmem.cuh"
#include <vector>
#include "morton.cuh"
#include "top.cuh"
#include "dumps.cuh"
#include "MortonHostModel.cuh"

namespace TST_ShiftReduce{
// -------------------------------------------------------------------------------------------------------------
int test01(){
	std::vector<uint64_t> vin = MortonHostModel::fillrnd_1bit(32);
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
// -------------------------------------------------------------------------------------------------------------
int tst_top(unsigned seed = 0){
	srand(seed ? seed : (unsigned)time(0));

	std::vector<uint64_t> vin = MortonHostModel::fillrnd_1bit(8);
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
// -------------------------------------------------------------------------------------------------------------
int tst_rnd_up(){

	for(int j = 0; j < 10000; j++){
		std::vector<uint64_t> vin64 = MortonHostModel::fillrnd_1bit(8);
		std::vector<int> vini = MortonHostModel::unpack(vin64);

		std::vector<int> vbot = MortonHostModel::reduct(vini);
		std::vector<int> vmid = MortonHostModel::reduct(vbot);
		std::vector<uint64_t> vresh = MortonHostModel::pack(vmid);
		assert (vresh.size() == 1);
		uint64_t resh = vresh[0];

		__half2* pout = nullptr;
		uint64_t mid = 0, bot = 0;
		uint64_t resd1 = get_top(vin64[0], pout, bot, mid);
		uint64_t resd2 = get_top(vin64[0], pout);

		if(resh != resd1 || resh != resd2){
			printf("tst_rnd_up() j=%d Error!\n", j);
			return -1;
		}
	}
	printf("tst_rnd_up() Ok\n");
	return 0;
}
// -------------------------------------------------------------------------------------------------------------
} // namespace TST_ShiftReduce{