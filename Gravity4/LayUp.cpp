#include "LayUp.h"
#include <cassert>

HostModel::LayUp::LayUp(size_t sz_side_){
	sz_side = sz_side_;
	v.resize(sz_side * sz_side);
}

HostModel::LayUp::LayUp(const std::vector<int>& v_inp){
	sz_side = (size_t)sqrt(double(v_inp.size()));
	assert(sz_side * sz_side == v_inp.size());
	v = v_inp;
}

std::vector<int>* HostModel::LayUp::run(const std::vector<int>& v_in){
	assert(v.size() * 4 == v_in.size());
	for(size_t j = 0; j < v.size(); j++){
		int sum = v_in[j * 4] + v_in[j * 4 + 1] + v_in[j * 4 + 2] + v_in[j * 4 + 3];
		v[j] = sum < 2 ? 0 : 1;
	}
	return &v;
}