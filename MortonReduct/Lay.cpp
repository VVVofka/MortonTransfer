#include "Lay.h"
#include <cassert>
using std::vector;
namespace LAYs{

void Lay::create(size_t N_, std::vector<int>* p_vaup, double k_lay, const double* p_kf){
	N = N_;
	pvaup = p_vaup;
	kLay = k_lay;
	sz_dn = 2ull << N;
	va_dn = vector<int>(sz_dn * sz_dn);
	vf_dn = vector<double>(sz_dn * sz_dn);
	for(int j = 0; j < 64; j++)
		pkF[j] = p_kf[j];
	//printf("N:%zu szSide:%zu szDn:%zu\n", N, sz_dn, va_dn.size());
} // ---------------------------------------------------------------------------------------------
std::vector<int>* Lay::load(const std::vector<int>& pdata_in){
	assert(pdata_in.size() == va_dn.size());
	va_dn = pdata_in;
	return &va_dn;
} // ---------------------------------------------------------------------------------------------
void Lay::run_up(){
	if(pvaup == nullptr) return;
	for(size_t j_up = 0; j_up < pvaup->size(); j_up++){
		const int* pdn = va_dn.data() + j_up * 4;
		int sum = pdn[0] + pdn[1] + pdn[2] + pdn[3];
		int val = sum < 2 ? 0 : 1;
		(*pvaup)[j_up] = val;
	}
} // ---------------------------------------------------------------------------------------------
vector<double>* Lay::run_dn(const vector<double>* pvfup, int dump_up/*- 1*/){
	if(pvfup){
		for(size_t j_up = 0; j_up < pvfup->size(); j_up++){
			const int* padn = va_dn.data() + j_up * 4;
			const size_t mask = padn[0] + (padn[1] << 1) + (padn[2] << 2) + (padn[3] << 3);
			const double* pkf4 = pkF + mask * 4;
			const double fup = (*pvfup)[j_up];
			for(size_t n = 0; n < 4; n++)
				vf_dn[j_up * 4 + n] = fup + pkf4[n] * kLay;
			if(dump_up == j_up){
				printf("%7d %zu%zu%zu%zu fup:%.2f + kF:%+.2f %+.2f %+.2f %+.2f * kLay:%.2f\n",
					dump_up * 4, mask >> 3, (mask >> 2) & 1, (mask >> 1) & 1, mask & 1,
					fup, pkf4[3], pkf4[2], pkf4[1], pkf4[0], kLay);
				printf("                  Result:   %+.2f %+.2f %+.2f %+.2f\n",
					vf_dn[j_up * 4 + 3], vf_dn[j_up * 4 + 2], vf_dn[j_up * 4 + 1], vf_dn[j_up * 4]);
			}
			const bool isdumplast = dump_a_last && ((j_up + 1) == pvfup->size());
			if(isdumplast){
				printf("%7zu %zu%zu%zu%zu fup:%.2f + kF:%+.2f %+.2f %+.2f %+.2f * kLay:%.2f\n",
					j_up * 4, mask >> 3, (mask >> 2) & 1, (mask >> 1) & 1, mask & 1,
					fup, pkf4[3], pkf4[2], pkf4[1], pkf4[0], kLay);
				printf("   Result pkf4[i] * kLay:   %+.2f %+.2f %+.2f %+.2f\n",
					pkf4[3] * kLay, pkf4[2] * kLay, pkf4[1] * kLay, pkf4[0] * kLay);
			}
		}
	} else{
		const size_t mask = va_dn[0] + (va_dn[1] << 1) + (va_dn[2] << 2) + (va_dn[3] << 3);
		const double* pkf4 = pkF + mask * 4;
		for(size_t n = 0; n < 4; n++)
			vf_dn[n] = pkf4[n] * kLay;
	}
	return &vf_dn;
} // ---------------------------------------------------------------------------------------------
void Lay::dump_va(){
	printf("%s", sdump_va().c_str());
} // ---------------------------------------------------------------------------------------------
std::string Lay::sdump_va(){
	std::string sret;
	for(size_t j = 0; j < va_dn.size(); j++){
		if(j && (j % 4) == 0) sret += " ";
		if(j && (j % 16) == 0) sret += " ";
		if(j && (j % 64) == 0) sret += "\n";
		if(j && (j % 256) == 0) sret += "\n";
		if(va_dn[j]) sret += "1";
		else sret += ".";
	}
	return sret + "\n";
} // ---------------------------------------------------------------------------------------------
}