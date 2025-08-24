#pragma once
#include <vector>
namespace HostModel{

class LayUp{
public:
	LayUp(size_t sz_side_ = 0){ sz_side = sz_side_; v.resize(sz_side * sz_side);}
	LayUp(const std::vector<int>& v_inp);

	std::vector<int> v;
	size_t sz_side = 0;
};
} 