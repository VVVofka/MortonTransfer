#pragma once
#include <vector>
#include <cuda_fp16.h>

namespace HostModel{

class LayDn{
public:
	LayDn(size_t sz_side = 0);
	
	size_t sz_side = 0;
	std::vector<__half2> v;
};
}
