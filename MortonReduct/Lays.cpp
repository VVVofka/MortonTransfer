#include "Lays.h"
using std::vector;
void Lays::create(int cnt_lays){
	vlays = std::vector<Lay>(cnt_lays);
	vector<int>* pnext = nullptr;
	for(size_t j = 0; j < vlays.size(); j++){
		Lay& lay = vlays[j];
		lay.create(pnext);
		pnext = &lay.va_dn;
	}
}
