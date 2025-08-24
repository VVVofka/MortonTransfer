#include "HostModel.h"
#include <cassert>
void HostModel::Model::create(size_t sz_side){
	vlay_up.clear();
	vlay_dn.clear();
	size_t sz_cur = sz0 = sz_side;
	while(sz_cur > 0){
		vlay_up.push_back(LayUp::LayUp(sz_cur));
		vlay_dn.push_back(LayDn::LayDn(sz_cur));
		sz_cur /= 2;
	}
}

void HostModel::Model::run(const std::vector<int>& v_in){
	for(size_t nlay = 0; nlay < vlay_up.size(); nlay++){

	}
}
