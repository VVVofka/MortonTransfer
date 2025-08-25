#pragma once
#include <vector>
namespace HostModel{

class LayUp{
public:
	LayUp(size_t sz_side_ = 0);
	LayUp(const std::vector<int>& v_inp);

	std::vector<int>* run(const std::vector<int>& v_inp); // return &v

	std::vector<int> v;
	size_t sz_side = 0;
};
} 