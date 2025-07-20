#include <cuda_runtime.h>
#include <cuda.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

__device__ __host__ __forceinline__ void reduct1bit(const uint64_t* src, uint64_t& dst){
	// sum by 4 bit, if sum < 2 then 0 else 1
	uint64_t sum = (src[0] & 1) + ((src[0] << 1) & 1) + ((src[0] << 2) & 1) + ((src[0] << 3) & 1);
	// 1 if >=2 else 0
	sum >>= 1;  // 1 if 2 or 3
	sum |= sum >> 1;    // or 1 if 4 ( res in pos 0)
	//compact by mask: 0b0001'0001'0001'0001''0001'0001'0001'0001''0001'0001'0001'0001''0001'0001'0001'0001;
	dst =
		((sum & 0x1000'0000'0000'0000) >> 45) | ((sum & 0x100'0000'0000'0000) >> 42) | ((sum & 0x10'0000'0000'0000) >> 39) | ((sum & 0x1'0000'0000'0000) >> 36) |
		((sum & 0x1000'0000'0000) >> 33) | ((sum & 0x100'0000'0000) >> 30) | ((sum & 0x10'0000'0000) >> 27) | ((sum & 0x1'0000'0000) >> 24) |
		((sum & 0x1000'0000) >> 21) | ((sum & 0x100'0000) >> 18) | ((sum & 0x10'0000) >> 15) | ((sum & 0x1'0000) >> 12) |
		((sum & 0x1000) >> 9) | ((sum & 0x100) >> 6) | ((sum & 0x10) >> 3) | (sum & 0x1);

	sum = (src[1] & 1) + ((src[1] << 1) & 1) + ((src[1] << 2) & 1) + ((src[1] << 3) & 1);
	sum |= sum >> 1;	// res in pos 0
	dst = sum & 0b0010'0010'0010'0010'0010'0010'0010'0010'0010'0010'0010'0010'0010'0010'0010'0010;

	sum = (src[2] & 1) + ((src[2] << 1) & 1) + ((src[2] << 2) & 1) + ((src[2] << 3) & 1);
	sum |= sum << 1;
	dst = sum & 0b0100'0100'0100'0100'0100'0100'0100'0100'0100'0100'0100'0100'0100'0100'0100'0100;

	sum = (src[3] & 1) + ((src[3] << 1) & 1) + ((src[3] << 2) & 1) + ((src[3] << 3) & 1);
	sum <<= 1;
	sum |= sum << 1;
	dst = sum & 0b1000'1000'1000'1000'1000'1000'1000'1000'1000'1000'1000'1000'1000'1000'1000'1000;
}