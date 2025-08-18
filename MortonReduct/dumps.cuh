#pragma once
// dumps.cuh
#include "common.cuh"
#include <valarray>
#include <vector>
#include <string>
#include "morton.cuh"
#include "CudaArray.h"
namespace Dumps{
// -------------------------------------------------------------------------------------------------------------
void dump2D_vhost(std::vector<uint64_t>& v, const std::string& caption = ""){
	printf("%s\n", caption.c_str());
	unsigned cntside = (unsigned)sqrt(double(v.size()));
	for(unsigned yr = 0; yr < cntside * 8; yr++){
		unsigned y = cntside * 8 - yr - 1;
		if((yr % 4) == 0 && yr){
			if((yr % 16) == 0)
				for(unsigned z = 0; z < cntside * 10 - 1; z++)
					printf("=");
			printf("\n");
		}
		for(unsigned x = 0; x < cntside * 8; x++){
			unsigned idmorton = EncodeMorton2(x, y);
			//printf("y:%u x:%u id_mortn=%u\n", y, x, idmorton);
			unsigned val = (v[idmorton / 64] >> (idmorton % 64)) & 1;
			if((x % 4) == 0 && x){
				if((x % 16) == 0)
					printf("|");
				else
					printf(" ");
			}
			if(val)
				printf("1");
			else
				printf(".");
		}
		printf("\n");
	}
}// -------------------------------------------------------------------------------------------------------------
void dump2D_cudaar(CudaArray<uint64_t>& cudaar, const std::string& caption = ""){
	std::vector<uint64_t> v = cudaar.get_vector();
	dump2D_vhost(v, caption);
}// -------------------------------------------------------------------------------------------------------------
void dump2D_uns64(uint64_t u, const std::string& caption = ""){
	std::vector<uint64_t> v{u};
	dump2D_vhost(v, caption);
}// -------------------------------------------------------------------------------------------------------------
template <typename T> void dump1D_uns64(T u, const std::string& caption = ""){
	if(caption != "") printf("%s", caption.c_str());
	for(T ir = 0; ir < sizeof(T) * 8; ir++){
		T i = sizeof(T) * 8 - ir - 1;
		if((ir % 4) == 0 && ir){
			if((ir % 16) == 0)
				printf("  ");
			else
				printf(" ");
		}
		if((u >> i) & 1)
			printf("1");
		else
			printf(".");
	}
	printf("\n");
}// -------------------------------------------------------------------------------------------------------------
template <typename T> void dump1D_uns64(std::vector<T> v, const std::string& caption = ""){
	if(caption != "") printf("%s", caption.c_str());
	for(int j=0; j<(int)v.size(); j++){
		dump1D_uns64<T>(v[j]);
	}
} // -------------------------------------------------------------------------------------------------------------
void dumpAr4(const std::valarray<double>& v, const std::string& caption = ""){
	printf("%s%+.2f %+.2f %+.2f %+.2f\n", caption.c_str(), v[0], v[1], v[2], v[3]);
 } // -------------------------------------------------------------------------------------------------------------
void VDouble(const std::vector<double>& v, size_t column=8, const std::string& caption = ""){
	if(caption != "") printf("%s\n", caption.c_str());
	for(size_t n = 0; n < v.size(); n++){
		printf("%+.2f ", v[n]);
		if(((n+1) % column == 0) && n)
			printf("\n");
	}
 } // -------------------------------------------------------------------------------------------------------------
void dump2Dxy(const std::vector<int>& v, const std::string& caption = ""){
	if(caption != "") printf("%s", caption.c_str());
	size_t side = (size_t) sqrt(double(v.size()));
	for(size_t y = 0; y < side; y++){
		for(size_t x = 0; x < side; x++){
			if((x % 4) == 0 && x){
				if((x % 16) == 0)
					printf("  ");
				else
					printf(" ");
			}
			auto val = v[y * side + x];
			if(val)
				printf("1");
			else
				printf(".");
		}
		printf("\n");
		if(y && (y % 4) == 0)
			printf("\n");
	}
 } // -------------------------------------------------------------------------------------------------------------
} // namespace Dumps::