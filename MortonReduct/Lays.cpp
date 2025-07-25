#include "Lays.h"
using std::vector;

vector<double>* Lay::pvKLays = nullptr;
vector<double>* Lay::pvkF = nullptr;
// ---------------------------------------------------------------------------------------------
void Lays::create(int cnt_lays){
	vlays = std::vector<Lay>(cnt_lays);
	vector<int>* pnext = nullptr;
	for(size_t j = 0; j < vlays.size(); j++){
		Lay& lay = vlays[j];
		lay.create(j, pnext);
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
