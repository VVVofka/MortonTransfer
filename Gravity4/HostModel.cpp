#include "HostModel.h"
#include <cassert>
void HostModel::Model::create(size_t sz_side){
	vlay_up.clear();
	size_t sz_cur = sz0 = sz_side;
	while((sz_cur /= 2) > 0){
		vlay_up.push_back(LayUp::LayUp(sz_cur));
	}

	vlay_dn.clear();
	sz_cur = 1;
	do{
		vlay_dn.push_back(LayDn::LayDn(sz_cur));
	} while((sz_cur *= 2) < sz0);
}

void HostModel::Model::run(std::vector<int>& v_in){
	std::vector<int>* pvin = & v_in;
	for(size_t nlay = 0; nlay < vlay_up.size(); nlay++){
		pvin = vlay_up[nlay].run(*pvin);
	}
}

__half2* HostModel::Model::fill_maskDn(){
	for(size_t j = 0; j < 16; j++){
		if(j == 0 || j == 15)
			maskDn[j] = 
	}
	return maskDn;
}
