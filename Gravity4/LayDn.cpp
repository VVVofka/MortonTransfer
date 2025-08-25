#include "LayDn.h"
#include <cassert>

HostModel::LayDn::LayDn(size_t sz_side_){
	sz_side = sz_side_;
	v = std::vector<__half2>(sz_side * sz_side, __float2half2_rn(0.f));
}

std::vector<__half2>* HostModel::LayDn::run(const std::vector<int>& v_a,
									const std::vector<__half2>& v_in,
									const __half2* pmask){
	assert(v.size() == v_in.size() * 4);
	assert(v_a.size() == v.size());
	for(size_t j = 0; j < v_in.size(); j++){
		int mask_a =	// v_a[x] 0 or 1 only
			v_a[j * 4] +
			v_a[j * 4 + 1] * 2 +
			v_a[j * 4 + 2] * 4 +
			v_a[j * 4 + 3] * 8;
		assert(mask_a < 16);
		const __half2* mask = pmask + mask_a * 4;
		for(int i = 0; i < 4; i++){
			v[j * 4 + i] = v_in[j] + mask[i];

		}
	}
	return &v;
}
