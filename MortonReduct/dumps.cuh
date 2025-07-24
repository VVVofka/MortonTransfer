#pragma once
// dumps.cuh
#include "common.cuh"

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
}
void dump2D_cudaar(CudaArray<uint64_t>& cudaar, const std::string& caption = ""){
	std::vector<uint64_t> v = cudaar.get_vector();
	dump2D_vhost(v, caption);
}
void dump2D_uns64(uint64_t u, const std::string& caption = ""){
	std::vector<uint64_t> v{u};
	dump2D_vhost(v, caption);
}
template <typename T>
void dump1D_uns64(T u, const std::string& caption = ""){
	printf("%s\n", caption.c_str());
	for(T ir = 0; ir < sizeof(T) * 8; ir++){
		T i = sizeof(T) * 8 - ir - 1;
		if((ir % 4) == 0 && ir){
			if((ir % 16) == 0)
				printf(" | ");
			else
				printf(" ");
		}
		if((u >> i) & 1)
			printf("1");
		else
			printf(".");
	}
	printf("\n");
}