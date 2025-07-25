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
vector<double>* Lay::run_dn(const vector<double>* pvfup){
	if(pvfup){
		for(size_t j_up = 0; j_up < pvfup->size(); j_up++){
			const int* padn = va_dn.data() + j_up * 4;
			const size_t mask = padn[0] + (padn[1] << 1) + (padn[2] << 2) + (padn[3] << 3);
			const double* pkf4 = pkF + mask * 4;
			const double fup = (*pvfup)[j_up];
			for(size_t n = 0; n < 4; n++)
				vf_dn[j_up * 4 + n] = fup + pkf4[n] * kLay;
		}
	} else{
		const size_t mask = va_dn[0] + (va_dn[1] << 1) + (va_dn[2] << 2) + (va_dn[3] << 3);
		const double* pkf4 = pkF + mask * 4;
		for(size_t n = 0; n < 4; n++)
			vf_dn[n] = pkf4[n] * kLay;
	}
	return &vf_dn;
} // ---------------------------------------------------------------------------------------------
}