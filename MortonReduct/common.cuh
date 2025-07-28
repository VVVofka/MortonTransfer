#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <cuda_fp16.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

uint32_t get_a(unsigned a4){return (0b1111111011101000 >> a4) & 1;}
uint32_t get_a15(unsigned a4){ return (0b1111111011101000 >> (a4 & 15)) & 1; }

namespace Convert{
template <typename T> std::vector<T> VectorHalf2ToVector(const std::vector<__half2>& vh2){
	std::vector<T> ret(vh2.size() * 2);
	const __half* p = reinterpret_cast<const __half*>(vh2.data());
	for(size_t j = 0; j < ret.size(); j++)
		ret[j] = T(p[j]);
	return ret;
} // --------------------------------------------------------------------------------------------------------------
template <typename T> std::vector<T> VectorHalf2ToVector(const __half2* pvh2, size_t sz){
	std::vector<T> ret(sz * 2);
	for(size_t j = 0; j < sz; j++){
		const __half2 h2 = pvh2[j];
		float x = __half2float(h2.x);
		float y = __half2float(h2.y);
		ret[j * 2] = T(x);
		ret[j * 2 + 1] = T(y);
	}
	return ret;
} // --------------------------------------------------------------------------------------------------------------
}
namespace Compare{
template <typename TA, typename TB>
bool vectors(const std::vector<TA>& va, const std::vector<TB>& vb, double epsilon = 0.0001){
	if(va.size() != vb.size()){
		printf("Different sizes!\n");
		return false;
	}
	for(size_t j = 0; j < va.size(); j++){
		double a = double(va[j]);
		double b = double(vb[j]);
		if(abs(a - b) > epsilon){
			printf("Different items %zu! a:%f b:%f\n", j, a, b);
			return false;
		}
	}
	return true;
}
}