#include "common.h"
#include <vector>

__device__ __host__ __forceinline__ uint64_t reduct64by1bit(const uint64_t* __restrict__ src){
	// sum by 4 bit, if sum < 2 then 0 else 1
	constexpr uint64_t M = 0x1111'1111'1111'1111ULL;
	uint64_t sum = (src[0] & M) + ((src[0] >> 1) & M) + ((src[0] >> 2) & M) + ((src[0] >> 3) & M);
	// 1 if >=2 else 0
	sum >>= 1;  // 1 if 2 or 3
	sum |= sum >> 1;    // or 1 if 4 ( res in pos 0)
	//compact by mask: 0b0001'0001'0001'0001''0001'0001'0001'0001''0001'0001'0001'0001''0001'0001'0001'0001;
	uint64_t dst =
		((sum & 0x1000'0000'0000'0000) >> 45) | ((sum & 0x100'0000'0000'0000) >> 42) | ((sum & 0x10'0000'0000'0000) >> 39) | ((sum & 0x1'0000'0000'0000) >> 36) |
		((sum & 0x1000'0000'0000) >> 33) | ((sum & 0x100'0000'0000) >> 30) | ((sum & 0x10'0000'0000) >> 27) | ((sum & 0x1'0000'0000) >> 24) |
		((sum & 0x1000'0000) >> 21) | ((sum & 0x100'0000) >> 18) | ((sum & 0x10'0000) >> 15) | ((sum & 0x1'0000) >> 12) |
		((sum & 0x1000) >> 9) | ((sum & 0x100) >> 6) | ((sum & 0x10) >> 3) | (sum & 0x1);

	sum = (src[1] & M) + ((src[1] >> 1) & M) + ((src[1] >> 2) & M) + ((src[1] >> 3) & M);
	sum |= sum >> 1;	// res in pos 1
	//compact by mask: 0b0010'0010'0010'0010''0010'0010'0010'0010''0010'0010'0010'0010''0010'0010'0010'0010;
	dst |=
		((sum & 0x2000'0000'0000'0000) >> 30) | ((sum & 0x200'0000'0000'0000) >> 27) | ((sum & 0x20'0000'0000'0000) >> 24) | ((sum & 0x2'0000'0000'0000) >> 21) |
		((sum & 0x2000'0000'0000) >> 18) | ((sum & 0x200'0000'0000) >> 15) | ((sum & 0x20'0000'0000) >> 12) | ((sum & 0x2'0000'0000) >> 9) |
		((sum & 0x2000'0000) >> 6) | ((sum & 0x200'0000) >> 3) | (sum & 0x20'0000) | ((sum & 0x2'0000) << 3) |
		((sum & 0x2000) << 6) | ((sum & 0x200) << 9) | ((sum & 0x20) << 12) | ((sum & 0x2) << 15);

	sum = (src[2] & M) + ((src[2] >> 1) & M) + ((src[2] >> 2) & M) + ((src[2] >> 3) & M);
	sum |= sum << 1;	// res in pos 2
	//compact by mask: 0b0100'0100'0100'0100''0100'0100'0100'0100''0100'0100'0100'0100''0100'0100'0100'0100;
	dst |=
		((sum & 0x4000'0000'0000'0000) >> 15) | ((sum & 0x400'0000'0000'0000) >> 12) | ((sum & 0x40'0000'0000'0000) >> 9) | ((sum & 0x4'0000'0000'0000) >> 6) |
		((sum & 0x4000'0000'0000) >> 3) | (sum & 0x400'0000'0000) | ((sum & 0x40'0000'0000) << 3) | ((sum & 0x4'0000'0000) << 6) |
		((sum & 0x4000'0000) << 9) | ((sum & 0x400'0000) << 12) | ((sum & 0x40'0000) << 15) | ((sum & 0x4'0000) << 18) |
		((sum & 0x4000) << 21) | ((sum & 0x400) << 24) | ((sum & 0x40) << 27) | ((sum & 0x4) << 30);

	sum = (src[3] & M) + ((src[3] >> 1) & M) + ((src[3] >> 2) & M) + ((src[3] >> 3) & M);
	sum <<= 1;
	sum |= sum << 1;// res in pos 3
	//compact by mask: 0b1000'1000'1000'1000''1000'1000'1000'1000''1000'1000'1000'1000''1000'1000'1000'1000;
	dst |=
		(sum & 0x8000'0000'0000'0000) | ((sum & 0x800'0000'0000'0000) << 3) | ((sum & 0x80'0000'0000'0000) << 6) | ((sum & 0x8'0000'0000'0000) << 9) |
		((sum & 0x8000'0000'0000) << 12) | ((sum & 0x800'0000'0000) << 15) | ((sum & 0x80'0000'0000) << 18) | ((sum & 0x8'0000'0000) << 21) |
		((sum & 0x8000'0000) << 24) | ((sum & 0x800'0000) << 27) | ((sum & 0x80'0000) << 30) | ((sum & 0x8'0000) << 33) |
		((sum & 0x8000) << 36) | ((sum & 0x800) << 39) | ((sum & 0x80) << 42) | ((sum & 0x8) << 45);
	return dst;
} // ////////////////////////////////////////////////////////////////////////////////
__host__ bool testreduct(unsigned seed){	// seed = 0
	if(seed) srand(seed);
	std::vector<int> vtstin(64 * 4);
	for(int& i : vtstin) i = rand() & 1;
	printf("vtstin:\n");
	for(int r = 0; r < 4; r++){
		for(int c = 0; c < 64; c++){
			if((c % 4 == 0) && c) printf(" ");
			if((c % 16 == 0) && c) printf(" ");
			printf("%d", vtstin[r * 64 + c]);
		}
		printf("\n");
	}
	printf("vtstres:\n");
	std::vector<int> vtstres(64);
	for(size_t j = 0; j < 64; j++){
		int sum = 0;
		for(size_t i = 0; i < 4; i++){
			sum += vtstin[j * 4 + i];
		}
		vtstres[j] = sum < 2 ? 0 : 1;
		if((j % 4 == 0) && j) printf(" ");
		if((j % 16 == 0) && j) printf("\n");
		printf("%d", vtstres[j]);
	}
	printf("\n");

	std::vector<uint64_t> vin64(4);
	for(size_t j = 0; j < 4; j++)
		for(uint64_t i = 0; i < 64; i++){
			size_t id = j * 64 + i;
			uint64_t added = uint64_t(vtstin[id]) << i;
			vin64[j] |= added;
		}
	printf("in64:\n");
	for(int j = 0; j < 4; j++) printf("%I64X\n", vin64[j]);

	uint64_t out64_old = reduct64by1bit(vin64.data());
	printf("old out64: %I64X\n", out64_old);

//	uint64_t out64_new = reduct1bit_wo_lut(vin64.data());
//	printf("new out64: %I64X\n", out64_new);

	for(int i = 0; i < 64; i++){
		int result_old = int((out64_old >> i) & 1);
		//int result_new = int((out64_new >> i) & 1);
		int expect = vtstres[i];
		if(result_old == expect /* && result_new == expect */) continue;
		if(result_old != expect)
			printf("Error old i=%d result=%d expect=%d\n", i, result_old, expect);
//		if(result_new != expect)
//			printf("Error new i=%d result=%d expect=%d\n", i, result_new, expect);
		return false;
	}
	printf("Test reduct64by1bit() Ok\n");
	return true;
}