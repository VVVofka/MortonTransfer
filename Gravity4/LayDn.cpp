#include "LayDn.h"

HostModel::LayDn::LayDn(size_t sz_side_){
	sz_side = sz_side_;
	v = std::vector<__half2>(sz_side * sz_side, __float2half2_rn(0.f));
}
