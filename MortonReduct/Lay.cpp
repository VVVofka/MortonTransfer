#include "Lay.h"
#include <cassert>
using std::vector;

void Lay::create(size_t N_, std::vector<int>* p_vaup){
	N = N_;
	pvaup = p_vaup;
	const size_t sz_dn = 4ull << (N * 2ull);
	va_dn = vector<int>(sz_dn);
	vf_dn = vector<double>(sz_dn);
} // ---------------------------------------------------------------------------------------------
std::vector<int>* Lay::load(const std::vector<int>* pdata_in){
	assert(pdata_in->size() == va_dn.size());
	va_dn = *pdata_in;
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
			const double* pkF = pvkF->data() + mask * 4;
			const double fup = (*pvfup)[j_up];
			for(size_t n = 0; n < 4; n++)
				vf_dn[j_up * 4 + n] = fup + pkF[n] * (*pvKLays)[N];
		}
	} else{
		const size_t mask = va_dn[0] + (va_dn[1] << 1) + (va_dn[2] << 2) + (va_dn[3] << 3);
		const double* pkF = pvkF->data() + mask * 4;
		for(size_t n = 0; n < 4; n++)
			vf_dn[n] = pkF[n] * (*pvKLays)[N];
	}
	return &vf_dn;
} // ---------------------------------------------------------------------------------------------
