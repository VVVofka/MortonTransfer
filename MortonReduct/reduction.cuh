#pragma once
//reduction.cuh
#include <cuda_runtime.h>
#include <vector>
#include "constmem.cuh"
#include "pack.cuh"
#include "convolution.cuh"


static __device__ __host__ __forceinline__ uint32_t reduct8by1bit(const uint32_t src){ // src 32 bit
	constexpr uint64_t M = 0x1111'1111U;
	uint64_t sum = (src & M) + ((src >> 1) & M) + ((src >> 2) & M) + ((src >> 3) & M);
	// 1 if >=2 else 0
	sum >>= 1;  // 1 if 2 or 3
	sum |= sum >> 1;    // or 1 if 4 ( res in pos 0)
	return 	// 8 bit
		((sum & 0x1000'0000) >> 21) | ((sum & 0x100'0000) >> 18) | ((sum & 0x10'0000) >> 15) | ((sum & 0x1'0000) >> 12) |
		((sum & 0x1000) >> 9) | ((sum & 0x100) >> 6) | ((sum & 0x10) >> 3) | (sum & 0x1);
} // ------------------------------------------------------------------------------------------------------------------
static __device__ __host__ __forceinline__ uint64_t reduct32by1bit(const uint64_t* __restrict__ src){ // src[2]
	// sum by 4 bit, if sum < 2 then 0 else 1
	constexpr uint64_t M = 0x1111'1111'1111'1111ULL;
	uint64_t sum = (src[0] & M) + ((src[0] >> 1) & M) + ((src[0] >> 2) & M) + ((src[0] >> 3) & M);
	// 1 if >=2 else 0
	sum >>= 1;  // 1 if 2 or 3
	sum |= sum >> 1;    // or 1 if 4 ( res in pos 0)
	//compact by mask: 0001'0001'0001'0001 0001'0001'0001'0001
	uint64_t dst =
		((sum & 0x1000'0000'0000'0000) >> 45) | ((sum & 0x100'0000'0000'0000) >> 42) | ((sum & 0x10'0000'0000'0000) >> 39) | ((sum & 0x1'0000'0000'0000) >> 36) |
		((sum & 0x1000'0000'0000) >> 33) | ((sum & 0x100'0000'0000) >> 30) | ((sum & 0x10'0000'0000) >> 27) | ((sum & 0x1'0000'0000) >> 24) |
		((sum & 0x1000'0000) >> 21) | ((sum & 0x100'0000) >> 18) | ((sum & 0x10'0000) >> 15) | ((sum & 0x1'0000) >> 12) |
		((sum & 0x1000) >> 9) | ((sum & 0x100) >> 6) | ((sum & 0x10) >> 3) | (sum & 0x1);
	return dst;	// 32 bit
} // ////////////////////////////////////////////////////////////////////////////////
static __device__ __host__ __inline__ uint64_t get_topA4A16(const uint64_t src){
	// 64 -> 16 unpack
	constexpr uint64_t M = 0x1111'1111'1111'1111ULL;
	uint64_t sum0 = (src & M) + ((src >> 1) & M) + ((src >> 2) & M) + ((src >> 3) & M);
	sum0 >>= 1;			// 1 if 2 or 3
	sum0 |= sum0 >> 1;  // or if 4 ( res in pos 0)
	uint64_t ret = pack16(sum0);

	// 16 unpack -> 4 unpack
	constexpr uint64_t M0 = 0x0001'0001'0001'0001ULL, M1 = M0 << 4, M2 = M1 << 4, M3 = M2 << 4;
	uint64_t sum1 = (sum0 & M0) + ((sum0 & M1) >> 4) + ((sum0 & M2) >> 8) + ((sum0 & M3) >> 12);
	sum1 >>= 1;			// 1 if 2 or 3
	sum1 |= sum1 >> 1;  // or if 4 ( res in pos 0)
	return (ret << 4) | pack4(sum1);
} // ----------------------------------------------------------------------------------------------
//static __device__ __host__ __inline__ uint64_t get_topA1A4A16(const uint64_t src){
//	// 64 -> 16 unpack
//	constexpr uint64_t M = 0x1111'1111'1111'1111ULL;
//	uint64_t sum0 = (src & M) + ((src >> 1) & M) + ((src >> 2) & M) + ((src >> 3) & M);
//	sum0 >>= 1;			// 1 if 2 or 3
//	sum0 |= sum0 >> 1;  // or if 4 ( res in pos 0)
//	uint64_t ret = pack16(sum0);
//
//	// 16 unpack -> 4 unpack
//	constexpr uint64_t M0 = 0x0001'0001'0001'0001ULL, M1 = M0 << 4, M2 = M1 << 4, M3 = M2 << 4;
//	uint64_t sum1 = (sum0 & M0) + ((sum0 & M1) >> 4) + ((sum0 & M2) >> 8) + ((sum0 & M3) >> 12);
//	sum1 >>= 1;			// 1 if 2 or 3
//	sum1 |= sum1 >> 1;  // or if 4 ( res in pos 0)
//	uint64_t pck = pack4(sum1);
//	return (ret << 5) | (pck << 1) | (uint64_t)get_a(pck);
//} // ----------------------------------------------------------------------------------------------
static __device__ __host__ __forceinline__ uint64_t reduct64to16bit(const uint64_t src){ // src:64 values 8 1 bit
	// sum by 4 bit, if sum < 2 then 0 else 1
	constexpr uint64_t M = 0x1111'1111'1111'1111ULL;
	uint64_t sum = (src & M) + ((src >> 1) & M) + ((src >> 2) & M) + ((src >> 3) & M);
	// 1 if >=2 else 0
	sum >>= 1;  // 1 if 2 or 3
	sum |= sum >> 1;    // or 1 if 4 ( res in pos 0)
	//compact 0x1111'1111'1111'1111 to 0xFFFF 
	uint64_t dst =
		((sum & 0x1000'0000'0000'0000) >> 45) | ((sum & 0x100'0000'0000'0000) >> 42) | ((sum & 0x10'0000'0000'0000) >> 39) | ((sum & 0x1'0000'0000'0000) >> 36) |
		((sum & 0x1000'0000'0000) >> 33) | ((sum & 0x100'0000'0000) >> 30) | ((sum & 0x10'0000'0000) >> 27) | ((sum & 0x1'0000'0000) >> 24) |
		((sum & 0x1000'0000) >> 21) | ((sum & 0x100'0000) >> 18) | ((sum & 0x10'0000) >> 15) | ((sum & 0x1'0000) >> 12) |
		((sum & 0x1000) >> 9) | ((sum & 0x100) >> 6) | ((sum & 0x10) >> 3) | (sum & 0x1);
	return dst;	// 32 bit
} // ////////////////////////////////////////////////////////////////////////////////
static __device__ __host__ __forceinline__ uint64_t reduct64to1(const uint64_t src){
	// sum by 4 bit, if sum < 2 then 0 else 1
	constexpr uint64_t M = 0x1111'1111'1111'1111ULL;
	uint64_t sum = (src & M) + ((src >> 1) & M) + ((src >> 2) & M) + ((src >> 3) & M);
	// 1 if >=2 else 0
	sum >>= 1;  // 1 if 2 or 3
	sum |= sum >> 1;    // or 1 if 4 ( res in pos 0)

	constexpr uint64_t M4 = 0x0001'0001'0001'0001ULL;
	sum = (sum & M4) | ((sum & (M4 << 4)) >> 3) | ((sum & (M4 << 8)) >> 6) | ((sum & (M4 << 12)) >> 9);
	sum = (sum & M4) + ((sum >> 1) & M4) + ((sum >> 2) & M4) + ((sum >> 3) & M4);
	sum >>= 1;  // 1 if 2 or 3
	sum |= sum >> 1;    // or 1 if 4 ( res in pos 0)

	constexpr uint64_t M1 = 1ULL;
	sum = (sum & M1) + ((sum >> 16) & M1) + ((sum >> 32) & M1) + ((sum >> 48) & M1);
	sum >>= 1;  // 1 if 2 or 3
	sum |= sum >> 1;    // or 1 if 4 ( res in pos 0)
	return sum & M1;
} // ////////////////////////////////////////////////////////////////////////////////
static __device__ __host__ __forceinline__ uint32_t reduct64to1_u32(const uint64_t src){
	// sum by 4 bit, if sum < 2 then 0 else 1
	constexpr uint64_t M = 0x1111'1111'1111'1111ULL;
	uint64_t sum = (src & M) + ((src >> 1) & M) + ((src >> 2) & M) + ((src >> 3) & M);
	// 1 if >=2 else 0
	sum >>= 1;  // 1 if 2 or 3
	sum |= sum >> 1;    // or 1 if 4 ( res in pos 0)

	constexpr uint64_t M4 = 0x0001'0001'0001'0001ULL;
	sum = (sum & M4) | ((sum & (M4 << 4)) >> 3) | ((sum & (M4 << 8)) >> 6) | ((sum & (M4 << 12)) >> 9);
	sum = (sum & M4) + ((sum >> 1) & M4) + ((sum >> 2) & M4) + ((sum >> 3) & M4);
	sum >>= 1;  // 1 if 2 or 3
	sum |= sum >> 1;    // or 1 if 4 ( res in pos 0)

	uint32_t ret = uint32_t((sum & 1) + ((sum >> 16) & 1) + ((sum >> 32) & 1) + ((sum >> 48) & 1));
	ret >>= 1;  // 1 if 2 or 3
	ret |= ret >> 1;    // or 1 if 4 ( res in pos 0)
	return ret & 1;
} // ////////////////////////////////////////////////////////////////////////////////
static __device__ __host__ __forceinline__ uint64_t reduct64by1bit(const uint64_t* __restrict__ src){
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
// ѕонижает длину стороны в 4 раза (площадь в 16)
static __device__ __inline__ void reduct64by1bit_x2(const uint64_t* __restrict__ data_dn,
							uint64_t* __restrict__ data_mid,
							uint64_t* __restrict__ data_up){
	const unsigned up_id_word = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned mid_id_word = up_id_word * 4;
#pragma unroll
	for(unsigned i = 0; i < 4; ++i){	// by dn word
		const unsigned cur_mid_id_word = mid_id_word + i;
		data_mid[cur_mid_id_word] = reduct64by1bit(data_dn + cur_mid_id_word * 4);
	}
	data_up[up_id_word] = reduct64by1bit(data_mid + mid_id_word);
}// ************************************************************************************
static __device__ __inline__ void reduct64by1bit_x2(const uint64_t* __restrict__ data_dn,
							uint64_t* __restrict__ data_up){
	uint64_t data_mid[4];
	reduct64by1bit_x2(data_dn, data_mid, data_up);
}// ************************************************************************************
__device__ __inline__ void reduct64toCombo(const uint64_t* __restrict__ data_in,
							uint32_t* __restrict__ data_out){


}// ************************************************************************************

static __host__ int testreduct(unsigned seed = 0){	// seed = 0
	srand(seed ? seed : (unsigned)time(0));
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
		return -1;
	}
	printf("Test reduct64by1bit() Ok\n");
	return 0;
} // ----------------------------------------------------------------------------------------------

//blockDim.x = 8
// psrc[8]
static __device__ __forceinline__
void reduct16for64bit_maxThread(const uint64_t* __restrict__ psrc, uint32_t* __restrict__ pdst){
	__shared__ uint32_t mid[4];
	const unsigned id_in = blockIdx.x * 8 + threadIdx.x;
	uint32_t tmp = uint32_t(pack16(convolution64to16(psrc[id_in])));
	uint32_t& m = mid[threadIdx.x / 2];
	m = threadIdx.x & 1 ? 0 : tmp;
	syncwarp();
	if(threadIdx.x & 1)
		m |= tmp << 16;
	syncwarp();

	if((threadIdx.x & 1) == 0)
		m = pack8(convolution32to8(m));
	syncwarp();

	if(threadIdx.x == 0)
		pack128to32(mid, pdst);
} // ----------------------------------------------------------------------------------------------
