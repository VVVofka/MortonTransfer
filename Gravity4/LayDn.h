#pragma once
#include <vector>
#include <cuda_fp16.h>

namespace HostModel{

class LayDn{
public:
	LayDn(size_t sz_side = 0);

	std::vector<__half2>* run(const std::vector<int>& v_a, 
						const std::vector<__half2>& v_up, 
						const __half2* pmask);
	
	size_t sz_side = 0;
	std::vector<__half2> v;
	
};
}
