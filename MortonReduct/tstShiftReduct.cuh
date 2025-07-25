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
#include <array>
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
	Dumps::dump2D_cudaar(lay_in, "IN");

	//std::string capt_shift0 = "SHIFT before " + std::to_string(shift.x) + "*" + std::to_string(shift.y);
	//dump_cudaar(lay_shift, capt_shift);

	glShiftReduction62by1X4_mid << <1, 1 >> > (lay_in.pdevice, lay_shift.pdevice, lay_mid.pdevice, lay_top.pdevice, shift);
	CHECK_CUDA(cudaDeviceSynchronize());

	std::string capt_shift1 = "\nSHIFT after " + std::to_string(shift.x) + "*" + std::to_string(shift.y);
	Dumps::dump2D_cudaar(lay_shift, capt_shift1);

	Dumps::dump2D_cudaar(lay_mid, "MID");
	Dumps::dump2D_cudaar(lay_top, "TOP");
	return 0;
}// -------------------------------------------------------------------------------------------------------------
int tst_rnd_up(){
	for(int j = 0; j < 10000; j++){
		std::vector<uint64_t> vin64 = MortonHostModel::fillrnd_1bit(8);
		std::vector<int> vini = MortonHostModel::unpack(vin64);

		std::vector<int> v16 = MortonHostModel::reduct(vini);
		std::vector<int> v4 = MortonHostModel::reduct(v16);
		std::vector<uint64_t> vresh = MortonHostModel::pack(v4);
		assert(vresh.size() == 1);
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
int up_f(){
	std::vector<uint64_t> vin64 = MortonHostModel::fillrnd_1bit(8);
	std::vector<int> vini = MortonHostModel::unpack(vin64);

	std::vector<int> v16 = MortonHostModel::reduct(vini);
	std::vector<int> v4 = MortonHostModel::reduct(v16);
	std::vector<uint64_t> vresh = MortonHostModel::pack(v4);
	assert(vresh.size() == 1);
	uint64_t resh4 = vresh[0];

	uint64_t in64 = vin64[0], resd4, resd16;
	get_topA4A16(in64, resd4, resd16);
	std::vector<int> vres16unpack = MortonHostModel::unpack(std::vector<uint64_t>({resd16}));
	vres16unpack.resize(v16.size());

	std::vector<MortonHostModel::Ar4> vklays(sizeof(MortonHostModel::kLay) / sizeof(MortonHostModel::kLay[0]));
	for(size_t j = 0; j < vklays.size(); j++)
		vklays[j] = MortonHostModel::Ar4(MortonHostModel::kLay[j], 4);

	Dumps::dump1D_uns64(resd4, "Top ");
	Dumps::dump1D_uns64(resd16, "Sec ");
	Dumps::dump1D_uns64(in64, "In: ");

	MortonHostModel::Ar4 kf_0 = MortonHostModel::kF(resd4);
	printf("kLay0: %.2f\n", vklays[0][0]);
	Dumps::dumpAr4(kf_0, "kf_0: ");
	MortonHostModel::Ar4 f_0 = kf_0 * vklays[0];
	Dumps::dumpAr4(f_0, "f_0=kf_0*kLay0: ");

	MortonHostModel::Ar4 f_1[4];
	MortonHostModel::Ar4 f_2[16];
	std::vector<double> f_3(64);
	printf("\nkLay1: %.2f\n", vklays[1][0]);
	for(int j = 0; j < 4; j++){
		printf("j = %d\n", j);
		uint64_t a1 = (resd16 >> (j * 4)) & 0xF;
		Dumps::dump1D_uns64(a1, "a1: ");

		MortonHostModel::Ar4 kf_1 = MortonHostModel::kF(a1);
		Dumps::dumpAr4(kf_1, "kf_1: ");
		MortonHostModel::Ar4 fup = MortonHostModel::Ar4(f_0[j], 4);
		printf("fup = %.2f\n", fup[0]);
		f_1[j] = kf_1 * vklays[1] + fup;
		Dumps::dumpAr4(f_1[j], "f_1[j] = kf_1 * vklays[1] + fup: ");
		for(int i = 0; i < 4; i++){
			printf("i = %d\n", i);
			uint64_t a2 = (in64 >> (j * 16 + i * 4)) & 0xF;
			Dumps::dump1D_uns64(a2, "a2: ");

			MortonHostModel::Ar4 kf_2 = MortonHostModel::kF(a2);
			Dumps::dumpAr4(kf_2, "kf_2: ");
			MortonHostModel::Ar4 fup_2 = MortonHostModel::Ar4(f_1[j][i], 4);
			printf("fup = %.2f\n", fup_2[0]);
			f_2[j * 4 + i] = kf_2 * vklays[2] + fup_2;
			Dumps::dumpAr4(f_2[j * 4 + i], "f_1[j*4+i] = kf_2 * vklays[2] + fup_2: ");
			for(int k = 0; k < 4; k++)
				f_3[j * 16 + i * 4 + k] = f_2[j * 4 + i][k];
		}
	}

	// device #########################################################
	CudaArray<uint64_t> lay_in(vin64);
	setSZ0toConstantMem(8);

	int2 shift{-8,-4};
	Dumps::dump2D_cudaar(lay_in, "IN");

	//std::string capt_shift0 = "SHIFT before " + std::to_string(shift.x) + "*" + std::to_string(shift.y);
	//dump_cudaar(lay_shift, capt_shift);

	//glShiftReduction62by1X4_mid << <1, 1 >> > (lay_in.pdevice, lay_shift.pdevice, lay_mid.pdevice, lay_top.pdevice, shift);
	//CHECK_CUDA(cudaDeviceSynchronize());

	//std::string capt_shift1 = "\nSHIFT after " + std::to_string(shift.x) + "*" + std::to_string(shift.y);
	//Dumps::dump2D_cudaar(lay_shift, capt_shift1);

	//Dumps::dump2D_cudaar(lay_mid, "MID");
	//Dumps::dump2D_cudaar(lay_top, "TOP");

	return 0;
}// -------------------------------------------------------------------------------------------------------------
} // namespace TST_ShiftReduce{