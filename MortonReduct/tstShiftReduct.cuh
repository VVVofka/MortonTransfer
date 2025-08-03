#pragma once
// tstShiftReduct.cuh
#include "common.cuh"
#include "CudaArray.h"
#include "CudaArrayD1.h"
#include "constmem.cuh"
#include <vector>
#include "morton.cuh"
#include "top.cuh"
#include "dumps.cuh"
#include "MortonHostModel.cuh"
#include <array>
#include <iostream>
#include "Lays.h"
using std::vector;
using std::string;
namespace TST_ShiftReduce{
// -------------------------------------------------------------------------------------------------------------
int test01(){
	std::vector<uint64_t> vin = MortonHostModel::fillrnd_1bit(32 * 32);
	CudaArray<uint64_t> lay_in(vin);
	CudaArray<uint64_t> lay_shift(4);
	CudaArray<uint64_t> lay_mid(2);
	CudaArray<uint64_t> lay_top(1);

	ConstMem::setSZ0(32);

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
		std::vector<uint64_t> vin64 = MortonHostModel::fillrnd_1bit(64);
		std::vector<int> vini = MortonHostModel::unpack(vin64);

		std::vector<int> v16 = MortonHostModel::reduct(vini);
		std::vector<int> v4 = MortonHostModel::reduct(v16);
		std::vector<uint64_t> vresh = MortonHostModel::pack(v4);
		assert(vresh.size() == 1);
		uint64_t resh4 = vresh[0];

		uint64_t a4_16;
		a4_16 = get_topA4A16(vin64[0]);
		std::vector<int> vres16unpack = MortonHostModel::unpack(std::vector<uint64_t>({a4_16 >> 4}));
		vres16unpack.resize(v16.size());

		if(resh4 != (a4_16 & 4)){
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
int up_f3(unsigned seed = 0){
	srand(seed ? seed : (unsigned)time(0));
	std::vector<uint64_t> vin64 = MortonHostModel::fillrnd_1bit(64);
	std::vector<int> vini = MortonHostModel::unpack(vin64);

	std::vector<int> v16 = MortonHostModel::reduct(vini);
	std::vector<int> v4 = MortonHostModel::reduct(v16);
	std::vector<uint64_t> vresh = MortonHostModel::pack(v4);
	assert(vresh.size() == 1);
	uint64_t resh4 = vresh[0];

	uint64_t in64 = vin64[0], a4_16;
	a4_16 = get_topA4A16(in64);
	std::vector<int> vres16unpack = MortonHostModel::unpack(std::vector<uint64_t>({a4_16 >> 4}));
	vres16unpack.resize(v16.size());

	std::vector<MortonHostModel::Ar4> vklays(sizeof(MortonHostModel::kLay) / sizeof(MortonHostModel::kLay[0]));
	for(size_t j = 0; j < vklays.size(); j++)
		vklays[j] = MortonHostModel::Ar4(MortonHostModel::kLay[j], 4);

	//Dumps::dump1D_uns64(resd4, "Top ");	printf("0x%016llX %zu\n", resd4, resd4);
	//Dumps::dump1D_uns64(resd16, "Sec ");	printf("0x%016llX %zu\n", resd16, resd16);
	//Dumps::dump1D_uns64(in64, "In: ");	printf("0x%016llX %zu\n", in64, in64);

	MortonHostModel::Ar4 kf_0 = MortonHostModel::kF(a4_16 & 15);
	//printf("kLay0: %.2f\n", vklays[0][0]);
	//Dumps::dumpAr4(kf_0, "kf_0: ");
	MortonHostModel::Ar4 f_0 = kf_0 * vklays[0];
	//Dumps::dumpAr4(f_0, "f_0=kf_0*kLay0: ");

	MortonHostModel::Ar4 f_1[4];
	MortonHostModel::Ar4 f_2[16];
	std::vector<double> vf_res_handwork(64);
	//printf("\nkLay1: %.2f\n", vklays[1][0]);
	for(int j = 0; j < 4; j++){
		//printf("\nj_lay1 = %d\n", j);
		uint64_t a1 = ((a4_16 >> 4) >> (j * 4)) & 0xF;
		//Dumps::dump1D_uns64(a1, "a1: ");		printf("a1: 0x%016llX %zu\n", a1, a1);

		MortonHostModel::Ar4 kf_1 = MortonHostModel::kF(a1);
		//Dumps::dumpAr4(kf_1, "kf_1: ");
		MortonHostModel::Ar4 fup = MortonHostModel::Ar4(f_0[j], 4);
		//printf("fup = %.2f\n", fup[0]);
		f_1[j] = kf_1 * vklays[1] + fup;
		//string sdump1 = "f_1[" + std::to_string(j) + "] = kf_1 * vklays[1] + fup: ";
		//Dumps::dumpAr4(f_1[j], sdump1);
		//printf("\nkLay2: %.2f\n", vklays[2][0]);
		for(int i = 0; i < 4; i++){
			//printf("i_lay2 = %d\n", i);
			uint64_t a2 = (in64 >> (j * 16 + i * 4)) & 0xF;
			//Dumps::dump1D_uns64(a2, "a2: ");			printf("a2: 0x%016llX %zu\n", a2, a2);

			MortonHostModel::Ar4 kf_2 = MortonHostModel::kF(a2);
			//Dumps::dumpAr4(kf_2, "kf_2: ");
			MortonHostModel::Ar4 fup_2 = MortonHostModel::Ar4(f_1[j][i], 4);
			//printf("fup_lay2 = %.2f\n", fup_2[0]);
			f_2[j * 4 + i] = kf_2 * vklays[2] + fup_2;
			//string sdump2 = "f_2[" + std::to_string(j) + "*4+" + std::to_string(i) + "] = kf_2 * vklays[2] + fup_2: ";
			//Dumps::dumpAr4(f_2[j * 4 + i], sdump2);
			for(size_t k = 0; k < 4; k++)
				vf_res_handwork[j * 16 + i * 4 + k] = f_2[j * 4 + i][k];
		}
	}
	//Dumps::VDouble(vf_res_handwork, 8, "vf_res_handwork:");
	// ####### Lays  ########################################################
	using namespace LAYs;
	Lays lays(3, MortonHostModel::kLay, MortonHostModel::vkF().data());
	vector<double> vf_res_lays = *lays.run(vini);
	//Dumps::VDouble(vf_res_lays, 8, "vf_res_lays:");
	if(Compare::vectors(vf_res_handwork, vf_res_lays) == false){
		return -1;
	}
	// device #########################################################
	CudaArray<uint64_t> data_a_in(vin64);
	CudaArrayD1<__half2> data_f_out(64 / 2);
	//ConstMem::setSZ0(8);
	ConstMem::loadKLay(MortonHostModel::kLay, sizeof(MortonHostModel::kLay) / sizeof(MortonHostModel::kLay[0]));
	ConstMem::loadKF4(MortonHostModel::vkF());
	//Dumps::dump2D_cudaar(data_a_in, "IN CUDA: ");

	glTop1 << <1, 32 >> > (data_a_in.pdevice, data_f_out.pdevice);
	CHECK_CUDA(cudaDeviceSynchronize());
	data_f_out.copy2host();
	vector<double> vhout = Convert::VectorHalf2ToVector<double>(data_f_out.phost, data_f_out.szall);
	//Dumps::VDouble(vhout, 8, "vhout:");
	if(Compare::vectors(vf_res_handwork, vhout, 0.003))
		return 0;
	return -1;
}// -------------------------------------------------------------------------------------------------------------
int up_f4(unsigned seed = 0){
	srand(seed ? seed : (unsigned)time(0));
	std::vector<uint64_t> vin64 = MortonHostModel::fillrnd_1bit(64 * 4);
	std::vector<int> vini = MortonHostModel::unpack(vin64);
	//for(int j = 0; j < 4; j++)Dumps::dump1D_uns64(vin64[j], "In: "),printf("0x%016llX %zu\n", vin64[j], vin64[j]);
	// ####### Lays  ########################################################
	using namespace LAYs;
	Lays lays(4, MortonHostModel::kLay, MortonHostModel::vkF().data());
	vector<double> vf_res_lays = *lays.run(vini, -3, -1);
	//Dumps::VDouble(vf_res_lays, 8, "vf_res_lays:");
	// device #########################################################
	CudaArray<uint64_t> data_a_in(vin64);
	CudaArrayD1<__half2> data_f_out(128);
	ConstMem::loadKLay(MortonHostModel::kLay, sizeof(MortonHostModel::kLay) / sizeof(MortonHostModel::kLay[0]));
	ConstMem::loadKF4(MortonHostModel::vkF());

	glTop2 << <1, 128 >> > (data_a_in.pdevice, data_f_out.pdevice);
	CHECK_CUDA(cudaDeviceSynchronize());
	data_f_out.copy2host();
	vector<double> vhout = Convert::VectorHalf2ToVector<double>(data_f_out.phost, data_f_out.szall);
	//Dumps::VDouble(vhout, 8, "vhout:");
	if(Compare::vectors(vf_res_lays, vhout, 0.003))
		return 0;
	return -1;
}// -------------------------------------------------------------------------------------------------------------
int up_f5(unsigned seed = 0){
	srand(seed ? seed : (unsigned)time(0));
	std::vector<uint64_t> vin64 = MortonHostModel::fillrnd_1bit(64 * 16);
	std::vector<int> vini = MortonHostModel::unpack(vin64);
	//for(int j = 0; j < 4; j++) Dumps::dump1D_uns64(vin64[j]), printf("0x%016llX  %zu\n", vin64[j], vin64[j]);
	// ####### Lays  ########################################################
	using namespace LAYs;
	Lays lays(5, MortonHostModel::kLay, MortonHostModel::vkF().data());
	vector<double> vf_res_lays = *lays.run(vini, -4, -256);
	//lays.dump();
	//Dumps::VDouble(vf_res_lays, 8, "vf_res_lays:");
	// device #########################################################
	CudaArray<uint64_t> data_a_in(vin64);
	CudaArrayD1<__half2> data_f_out(512);
	ConstMem::loadKLay(MortonHostModel::kLay, sizeof(MortonHostModel::kLay) / sizeof(MortonHostModel::kLay[0]));
	ConstMem::loadKF4(MortonHostModel::vkF());

	glTop3 << <1, 512 >> > (data_a_in.pdevice, data_f_out.pdevice);
	CHECK_CUDA(cudaDeviceSynchronize());
	data_f_out.copy2host();
	vector<double> vhout = Convert::VectorHalf2ToVector<double>(data_f_out.phost, data_f_out.szall);
	//Dumps::VDouble(vhout, 8, "vhout:");
	if(Compare::vectors(vf_res_lays, vhout, 0.003))
		return 0;
	return -1;
}// -------------------------------------------------------------------------------------------------------------

int mid(unsigned seed = 0){
	srand(seed ? seed : (unsigned)time(0));
	std::vector<uint64_t> vin64 = MortonHostModel::fillrnd_1bit(1024 * 1024, 0.45);
	std::vector<int> vini = MortonHostModel::unpack(vin64);
	{ auto i = vin64[vin64.size() - 1]; Dumps::dump1D_uns64(i, "Last "), printf("%zu 0x%016llX  %zu\n", vin64.size() - 1, i, i); }
	// ####### Lays  ########################################################
	using namespace LAYs;
	Lays lays(10, MortonHostModel::kLay, MortonHostModel::vkF().data());
	for(auto& i: lays.vlays) i.dump_a_last = true;
	vector<double> vf_res_lays = *lays.run(vini, -4, -256);
	lays.dump_a_first("Lays First:");
	//lays.dump_a_last("Lays Last:");
	// device #########################################################
	CudaArray<uint64_t> data_a_in(vin64);
	CudaArrayD1<__half2> data_f_in(512);
	CudaArrayD1<__half2> data_f_out(524288);
	CudaArrayD1<uint32_t> data_a_out(512);
	ConstMem::loadKLay(MortonHostModel::kLay, sizeof(MortonHostModel::kLay) / sizeof(MortonHostModel::kLay[0]));
	ConstMem::loadKF4(MortonHostModel::vkF());

	glUpMid3 << <512, 1024 >> > (data_a_in.pdevice, data_a_out.pdevice);
	//glDnMid3 << <512, 1024 >> > (data_f_in.pdevice, data_a_in.pdevice, data_f_out.pdevice);
	CHECK_CUDA(cudaDeviceSynchronize());
	data_a_out.copy2host();
	std::vector<uint64_t> voutamidup(16);
	size_t i = 0;
	for(size_t j = 0; j < voutamidup.size(); j++){
		for(size_t n = 0; n < 32; n++){
			voutamidup[j] |= (uint64_t)data_a_out.phost[i++] << (2 * n);
		}
	}
	//Dumps::dump1D_uns64(voutamidup, "device:\n");
	auto vhostpack = MortonHostModel::pack<uint64_t, int>(lays.vlays[5].va_dn);
	//Dumps::dump1D_uns64(vhostpack, "host:\n");

	data_f_out.copy2host();
	vector<double> vhout = Convert::VectorHalf2ToVector<double>(data_f_out.phost, data_f_out.szall);
	//Dumps::VDouble(vhout, 8, "vhout:");
	if(Compare::vectors(vf_res_lays, vhout, 0.003))
		return 0;
	return -1;
}// -------------------------------------------------------------------------------------------------------------

} // namespace TST_ShiftReduce{