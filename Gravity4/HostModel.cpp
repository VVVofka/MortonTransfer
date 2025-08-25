#include "HostModel.h"
#include <cassert>

HostModel::Model::Model(){
	fill_maskDn();
}

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

	assert(vlay_up.size() == vlay_dn.size());
	kLays.resize(vlay_dn.size());
	for(size_t nlay = 0; nlay < vlay_dn.size(); nlay++)
		kLays[nlay] = __half2(1.f, 1.f);
}

void HostModel::Model::run(std::vector<int>& v_in){
	run_up(v_in);
	run_dn(v_in, maskDn);
}

void HostModel::Model::run_up(std::vector<int>& v_in){
	std::vector<int>* pvin = &v_in;
	for(size_t nlay = 0; nlay < vlay_up.size(); nlay++){
		pvin = vlay_up[nlay].run(*pvin);
	}
}

std::vector<__half2>* HostModel::Model::run_dn(std::vector<int>& v_out, const __half2* pmask){
	std::vector<__half2>* pvin = &vlay_dn[0].v;
	for(size_t nlay = 0; nlay < vlay_up.size() - 1; nlay++){
		const LayUp& pva = vlay_up[vlay_up.size() - 2 - nlay];
		pvin = vlay_dn[nlay].run(pva.v, *pvin, pmask);
	}
	pvin = vlay_dn[vlay_dn.size() - 1].run(v_out, *pvin, pmask);
	return pvin;
}

__half2* HostModel::Model::fill_maskDn(){
	__half val0 = __half(0.);
	__half val07 = __half(1. / sqrt(2.));
	__half2* b = maskDn;
	for(size_t j = 0; j < 16; j++){
		if(j == 0 || j == 15){
			*b++ = __half2(-val07, -val07);
			*b++ = __half2(val07, -val07);
			*b++ = __half2(-val07, val07);
			*b++ = __half2(val07, val07);
		} else{
			*b++ = __half2(val0, val0);
			*b++ = __half2(val0, val0);
			*b++ = __half2(val0, val0);
			*b++ = __half2(val0, val0);
		}
	}
	return maskDn;
}
