#include "Lays.h"
#include <cassert>
using std::vector;
namespace LAYs{

// ---------------------------------------------------------------------------------------------
void Lays::create(int cnt_lays, const double* p_klay, const double* p_kf){
	vlays = std::vector<Lay>(cnt_lays);
	vector<int>* pa_up = nullptr;
	for(int nlay = 0; nlay < vlays.size(); nlay++){
		Lay& lay = vlays[nlay];
		lay.create(nlay, pa_up, p_klay[nlay], p_kf);
		pa_up = &lay.va_dn;
	}
} // ---------------------------------------------------------------------------------------------
std::vector<double>* Lays::run(const vector<int>& data_in, int dump_lay/*-1*/, int dump_j/*-1*/){
	assert(vlays.size());
	int nlay = (int)vlays.size() - 1;
	vlays[nlay].load(data_in);
	while(nlay >= 0){
		vlays[nlay].run_up();
		--nlay;
	}
	vector<double>* pfup = nullptr;
	while(++nlay < (int)vlays.size()){
		int ndump = (dump_lay == nlay) ? dump_j / 4 : -1;
		pfup = vlays[nlay].run_dn(pfup, ndump);
	}
	return pfup;
} // ---------------------------------------------------------------------------------------------
void Lays::dump(){
	for(int nlay = 0; nlay < (int)vlays.size(); nlay++){
		printf("\nLay %d\n", nlay);
		vlays[nlay].dump_va();
	}
} // ---------------------------------------------------------------------------------------------
void Lays::dump_lay(size_t n_lay){
	vlays[n_lay].dump_va();
} // ---------------------------------------------------------------------------------------------
}