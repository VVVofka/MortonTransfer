#include "Lays.h"
using std::vector;
namespace LAYs{

// ---------------------------------------------------------------------------------------------
void Lays::create(int cnt_lays, const double* p_klay, const double* p_kf){
	vlays = std::vector<Lay>(cnt_lays);
	vector<int>* pnext = nullptr;
	for(size_t nlay = 0; nlay < vlays.size(); nlay++){
		Lay& lay = vlays[nlay];
		lay.create(nlay, pnext, p_klay[nlay], p_kf);
		pnext = &lay.va_dn;
	}
} // ---------------------------------------------------------------------------------------------
std::vector<double>* Lays::run(const vector<int>* data_in){
	int nlay = 0;
	Lay& lay = vlays[nlay];
	lay.load(data_in);
	while(nlay < (int)vlays.size()){
		lay = vlays[nlay];
		lay.run_up();
		nlay++;
	}
	vector<double>* pfup = nullptr;
	while(--nlay >= 0){
		lay = vlays[nlay];
		pfup = lay.run_dn(pfup);
	}
	return pfup;
} // ---------------------------------------------------------------------------------------------
}