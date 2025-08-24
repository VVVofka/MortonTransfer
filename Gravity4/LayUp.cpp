#include "LayUp.h"
#include <cassert>
HostModel::LayUp::LayUp(const std::vector<int>& v_inp){
	sz_side = (size_t)sqrt(double(v_inp.size()));
	assert(sz_side * sz_side == v_inp.size());
	v = v_inp;
}