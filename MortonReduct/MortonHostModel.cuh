#pragma once
// MortonHostModel.cuh
#include "common.cuh"
#include "morton.cuh"
#include <valarray>
#include <array>
#include <cassert>

namespace MortonHostModel{
typedef	std::valarray<double> Ar4;
Ar4 kF(size_t n){
	assert(n < 16);
	return std::array<Ar4, 16>{
		Ar4{0.3, 0.3, 0.3, 0.3},	// 0000 0
		Ar4{-0.2, 0.1, 0.1, 0.1},	// 0001 4
		Ar4{0.1, -0.2, 0.1, 0.1},	// 0010 8
		Ar4{0.0, 0.0, 0.0, 0.0},	// 0011 12

		Ar4{0.1, 0.1, -0.2, 0.1},	// 0100 16 
		Ar4{0.0, 0.0, 0.0, 0.0},	// 0101 20
		Ar4{0.0, 0.0, 0.0, 0.0},	// 0110 24 
		Ar4{-0.1, -0.1, -0.1, -0.1},// 0111 28

		Ar4{0.1, 0.1, 0.1, -0.2},	// 1000 32
		Ar4{0.0, 0.0, 0.0, 0.0},	// 1001 36
		Ar4{0.0, 0.0, 0.0, 0.0},	// 1010 40
		Ar4{-0.1, -0.1, 0.2, -0.1},	// 1011 44

		Ar4{0.0, 0.0, 0.0, 0.0},	// 1100 48
		Ar4{-0.1, 0.2, -0.1, -0.1},	// 1101 52
		Ar4{0.2, -0.1, -0.1, -0.1},	// 1110 56
		Ar4{-1.0, -1.0, -1.0, -1.0}	// 1111 60
	}[n];
} // ---------------------------------------------------------------------------------------------
std::vector<double> vkF(){
	std::vector<double> ret(64);
	for(int j=0; j<64; j++)
		ret[j] = kF(j / 4)[j % 4];
	return ret;
} // ---------------------------------------------------------------------------------------------
double kLay[6]{0.8, 0.9, 1.0, 1.1, 1.2, 1.3};

typedef std::vector<int> ivector;
// -------------------------------------------------------------------------------------------------------------
std::vector<uint64_t> fillrnd_1bit(unsigned sz_all, double k = 0.3){
	const int level = int(255 * k);
	std::vector<uint64_t> ret(sz_all / 64);
	for(size_t nword = 0; nword < ret.size(); nword++){
		ret[nword] = 0;
		for(uint64_t j = 0; j < 64; j++){
			if((rand() & 0xFF) < level)
				ret[nword] |= 1ull << j;
		}
	}
	return ret;
}
// -------------------------------------------------------------------------------------------------------------
template <typename T_OUT = uint64_t, typename T_IN = int>
std::vector<T_OUT> pack(const std::vector<T_IN>& v){
	const auto cntbit = sizeof(T_OUT) * 8;
	size_t szret = v.size() / (4 * cntbit);
	if(szret == 0) szret = 1;
	std::vector<T_OUT> ret(szret);
	for(size_t iout = 0; iout < ret.size(); iout++){
		for(size_t ibit = 0; ibit < cntbit; ibit++){
			size_t i_in = iout * cntbit + ibit;
			if(i_in < v.size()){
				auto val = v[i_in] & 1;
				ret[iout] |= T_OUT(val) << ibit;
			}
		}
	}
	return ret;
}  // -------------------------------------------------------------------------------------------------------------
template <typename T = uint64_t>
ivector unpack(const std::vector<T>& v){
	const unsigned cntbit = sizeof(T) * 8;
	ivector ret(v.size() * cntbit);
	for(size_t j = 0; j < v.size(); j++)
		for(unsigned n = 0; n < cntbit; n++)
			ret[j * cntbit + n] = (v[j] >> n) & 1;
	return ret;
}  // -------------------------------------------------------------------------------------------------------------
template <typename T>
std::vector<T> reduct(const std::vector<T>& v){
	std::vector<T> ret(v.size() / 4);
	assert(v.size() && v.size() % 4 == 0);
	for(unsigned j = 0; j < ret.size(); j++){
		const T* p = v.data() + j * 4;
		const T sum = p[0] + p[1] + p[2] + p[3];
		ret[j] = sum < 2 ? 0 : 1;
	}
	return ret;
} // -------------------------------------------------------------------------------------------------------------
}