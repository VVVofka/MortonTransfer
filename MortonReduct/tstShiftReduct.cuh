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
}// -------------------------------------------------------------------------------------------------------------
int tst_rnd_up(){
	for(int j = 0; j < 10000; j++){
		std::vector<uint64_t> vin64 = MortonHostModel::fillrnd_1bit(8);
		std::vector<int> vini = MortonHostModel::unpack(vin64);

		std::vector<int> v16 = MortonHostModel::reduct(vini);
		std::vector<int> v4 = MortonHostModel::reduct(v16);
		std::vector<uint64_t> vresh = MortonHostModel::pack(v4);
		assert (vresh.size() == 1);
		uint64_t resh4 = vresh[0];

		uint64_t resd4, resd16;
		get_topA4A16(vin64[0], resd4, resd16);
		std::vector<int> vres16unpack = MortonHostModel::unpack(std::vector<uint64_t>({resd16}));
		vres16unpack.resize(v16.size());

		if(resh4 != resd4){
			printf("tst_rnd_up() j=%d Error! (resh4 != resd4)\n", j);
			return -1;
		}
		if(vres16unpack != v16){
			printf("tst_rnd_up() j=%d Error! (vres16unpack != v16)\n", j);
			return -2;
		}
	}
	printf("tst_rnd_up() Ok\n");
	return 0;
}// -------------------------------------------------------------------------------------------------------------
} // namespace TST_ShiftReduce{