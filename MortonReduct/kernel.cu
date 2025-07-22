#include "common.cuh"
//#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <chrono>
#include <string>
#include "reduction.cuh"
#include "morton.cuh"

__constant__ unsigned SZ0;


static void setSZ0toConstantMem(unsigned sz0){
	CHECK_CUDA(cudaMemcpyToSymbol(static_cast<const void*>(&SZ0), &sz0, sizeof(sz0), 0, cudaMemcpyHostToDevice));
}

static __device__ __host__ __forceinline__ unsigned get2bits64(const uint64_t* data, unsigned index){
	unsigned word_index = index >> 5;
	unsigned bit_offset = (index & 31) << 1;
	return (data[word_index] >> bit_offset) & 0x3;
}
static __device__ __host__ __forceinline__ unsigned get2bits32(const uint32_t* data, unsigned index){
	unsigned word_index = index >> 4;
	unsigned bit_offset = (index & 15) << 1;
	return (data[word_index] >> bit_offset) & 0x3;
}

static __device__ __host__ __forceinline__ void set2bits64(uint64_t* data, unsigned index, unsigned value){
	unsigned word_index = index >> 5;
	unsigned bit_offset = (index & 31) << 1;
	uint64_t mask = ~(0x3ULL << bit_offset);
	data[word_index] = (data[word_index] & mask) | ((uint64_t)(value & 0x3) << bit_offset);
}

static void dump2bit_grid(const std::vector<uint64_t>& data, const char* title = nullptr){
	const unsigned size_side = (unsigned)sqrt(double(32 * data.size()));
	if(title) std::cout << "--- " << title << " ---\n";
	for(unsigned y = 0; y < size_side; ++y){
		for(unsigned x = 0; x < size_side; ++x){
			unsigned idx_morton = EncodeMorton2h(x, y);
			unsigned val = get2bits64(data.data(), idx_morton);
			if(val)
				std::cout << val << " ";
			else
				std::cout << ". ";
		}
		std::cout << "\n";
	}
	std::cout << std::endl;
}// -------------------------------------------------------------------------
static void dump2bit_grid(const std::vector<uint32_t>& data, const char* title = nullptr){
	const unsigned size_side = (unsigned)sqrt(double(16 * data.size()));
	if(title) std::cout << "--- " << title << " ---\n";
	for(unsigned y = 0; y < size_side; ++y){
		for(unsigned x = 0; x < size_side; ++x){
			unsigned idx_morton = EncodeMorton2h(x, y);
			unsigned val = get2bits32(data.data(), idx_morton);
			if(val)
				std::cout << val << " ";
			else
				std::cout << ". ";
		}
		std::cout << "\n";
	}
	std::cout << std::endl;
}// -------------------------------------------------------------------------
// return 0 if count bit <2, else 1
__device__ __forceinline__ unsigned getMask_4to1bit(unsigned a4){
	return (0b1111'1110'1110'1000 >> (a4 & 15)) & 1;
}
__device__ __forceinline__ unsigned getMask_8to2bit(unsigned a8){
	// упаковываем чётные биты
	unsigned val0 = getMask_4to1bit(a8 & 1 | ((a8 >> 1) & 2) | ((a8 >> 2) & 4) | ((a8 >> 3) & 8));
	// упаковываем нечётные биты
	unsigned val1 = getMask_4to1bit(((a8 >> 1) & 1) | ((a8 >> 2) & 2) | ((a8 >> 3) & 4) | ((a8 >> 4) & 8));
	return val0 | (val1 << 1);
}// ***********************************************************************************************
__global__ void reduct(const uint64_t* __restrict__ data_in,
					   uint64_t* __restrict__ data_shift,
					   unsigned* __restrict__ data_mid,
					   unsigned* __restrict__ data_top,
					   const int2 shift){
	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x; // top-level index

	const unsigned top_side = SZ0 >> 2;  // 1/4 size
	const unsigned mid_side = SZ0 >> 1;  // 1/2 size

	unsigned top_x = DecodeMorton2X(tid);
	unsigned top_y = DecodeMorton2Y(tid);

	if(top_x >= top_side || top_y >= top_side) return;

	unsigned mid_base = (top_y * 2) * mid_side + (top_x * 2);
	unsigned in_base_x = top_x * 4;
	unsigned in_base_y = top_y * 4;

	// 1. Read and shift 16x16 block, store to data_shift
#pragma unroll
	for(unsigned j = 0; j < 4 * 4; ++j){
#pragma unroll
		for(unsigned i = 0; i < 4 * 4; ++i){
			unsigned x_in = (in_base_x + i - shift.x + SZ0) & (SZ0 - 1);
			unsigned y_in = (in_base_y + j - shift.y + SZ0) & (SZ0 - 1);
			unsigned in_idx = EncodeMorton2(x_in, y_in);
			unsigned val = get2bits64(data_in, in_idx);
			unsigned out_idx = EncodeMorton2(in_base_x + i, in_base_y + j);
			set2bits64(data_shift, out_idx, val);
		}
	}
	// 2. Reduce 16x16 block to 8x8 mid layer (2 bits -> 1 bit x2)
	unsigned combined = 0;
#pragma unroll
	for(unsigned my = 0; my < 2; ++my){
#pragma unroll
		for(unsigned mx = 0; mx < 2; ++mx){
			unsigned local_mask = 0;
#pragma unroll
			for(unsigned j = 0; j < 4; ++j){
#pragma unroll
				for(unsigned i = 0; i < 4; ++i){
					unsigned x = in_base_x + mx * 4 + i;
					unsigned y = in_base_y + my * 4 + j;
					unsigned idx = EncodeMorton2(x, y);
					unsigned val = get2bits64(data_shift, idx);
					local_mask |= (val & 1) << (j * 4 + i);
				}
			}
			unsigned mid_val = getMask_8to2bit(local_mask);
			data_mid[mid_base + my * mid_side + mx] = mid_val;
			combined |= (mid_val & 3) << ((my * 2 + mx) * 2);
		}
	}
	// 3. Reduce combined 8-bit mask to final top value
	data_top[tid] = combined;
} // **************************************************************************
__host__ uint32_t bit_reverse32(uint32_t x){
	x = ((x & 0x55555555) << 1) | ((x >> 1) & 0x55555555);
	x = ((x & 0x33333333) << 2) | ((x >> 2) & 0x33333333);
	x = ((x & 0x0F0F0F0F) << 4) | ((x >> 4) & 0x0F0F0F0F);
	x = ((x & 0x00FF00FF) << 8) | ((x >> 8) & 0x00FF00FF);
	return (x << 16) | (x >> 16);
}
__host__ uint64_t bit_reverse64(uint64_t x){
	x = ((x & 0x5555555555555555ULL) << 1) | ((x >> 1) & 0x5555555555555555ULL);
	x = ((x & 0x3333333333333333ULL) << 2) | ((x >> 2) & 0x3333333333333333ULL);
	x = ((x & 0x0F0F0F0F0F0F0F0FULL) << 4) | ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL);
	x = ((x & 0x00FF00FF00FF00FFULL) << 8) | ((x >> 8) & 0x00FF00FF00FF00FFULL);
	x = ((x & 0x0000FFFF0000FFFFULL) << 16) | ((x >> 16) & 0x0000FFFF0000FFFFULL);
	return (x << 32) | (x >> 32);
}
//int test_reduce(unsigned size_side = 32){
//	srand(42);
//	const unsigned total_values = size_side * size_side;
//	const unsigned total_words = (total_values + 31) / 32;
//
//	const unsigned mid_side = size_side >> 1;
//	const unsigned mid_len = mid_side * mid_side;
//	const unsigned total_words_mid = (mid_len + 15) / 16;
//
//	const unsigned top_side = size_side >> 2;
//	const unsigned top_len = top_side * top_side;
//	const unsigned total_words_top = (top_len + 15) / 16; // исправлено
//
//	std::vector<uint64_t> h_input(total_words, 0);
//	std::vector<uint64_t> h_shifted(total_words, 0);
//	std::vector<unsigned> h_mid(mid_len, 0);	// исправлено: размер именно mid_len
//	std::vector<unsigned> h_top(top_len, 0);	// исправлено: размер top_len
//
//	// 1. Fill input with pseudo-random pattern
//	for(unsigned i = 0; i < total_values; ++i){
//		unsigned rnd1 = ((double)rand() / RAND_MAX) < 0.73 ? 0 : 2;
//		unsigned rnd0 = ((double)rand() / RAND_MAX) < 0.77 ? 0 : 1;
//		unsigned val = rnd1 + rnd0;
//		unsigned word = i >> 5;
//		unsigned bit = (i & 31) << 1;
//		h_input[word] |= uint64_t(val & 0x3) << bit;
//	}
//	dump2bit_grid(h_input, "Input");
//
//	// 2. Compute expected results on CPU
//	auto get_val = [&](const std::vector<uint64_t>& d, unsigned idx) -> unsigned{
//		return (d[idx >> 5] >> ((idx & 31) << 1)) & 0x3;
//	};
//
//	auto bit_reverse32 = [](unsigned x) -> unsigned{
//		x = __builtin_bswap32(x);
//		x = ((x & 0x0F0F0F0F) << 4) | ((x & 0xF0F0F0F0) >> 4);
//		x = ((x & 0x33333333) << 2) | ((x & 0xCCCCCCCC) >> 2);
//		x = ((x & 0x55555555) << 1) | ((x & 0xAAAAAAAA) >> 1);
//		return x;
//	};
//
//	auto bit_reverse64 = [](uint64_t x) -> uint64_t{
//		x = __builtin_bswap64(x);
//		x = ((x & 0x0F0F0F0F0F0F0F0FULL) << 4) | ((x & 0xF0F0F0F0F0F0F0F0ULL) >> 4);
//		x = ((x & 0x3333333333333333ULL) << 2) | ((x & 0xCCCCCCCCCCCCCCCCULL) >> 2);
//		x = ((x & 0x5555555555555555ULL) << 1) | ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1);
//		return x;
//	};
//
//	auto encode = [&](unsigned x, unsigned y){
//		return bit_reverse64((bit_reverse32(x) >> 1) | ((bit_reverse32(y) >> 1) << 1)) >> 32;
//	};
//
//	auto mask4 = [](unsigned a4){
//		return (0b1111111011101000 >> (a4 & 15)) & 1;
//	};
//	auto mask8 = [&](unsigned a8){
//		return mask4(a8 & 0xF) | (mask4((a8 >> 1) & 0xF) << 1);
//	};
//
//	int2 shift = {3, -5};
//
//	for(unsigned ty = 0; ty < top_side; ++ty){
//		for(unsigned tx = 0; tx < top_side; ++tx){
//			unsigned in_base_x = tx * 4;
//			unsigned in_base_y = ty * 4;
//			unsigned tid = encode(tx, ty);
//
//			unsigned combined = 0;
//			for(unsigned my = 0; my < 2; ++my){
//				for(unsigned mx = 0; mx < 2; ++mx){
//					unsigned local_mask = 0;
//					for(unsigned j = 0; j < 4; ++j){
//						for(unsigned i = 0; i < 4; ++i){
//							unsigned x = (in_base_x + mx * 2 + i - shift.x + size_side) & (size_side - 1);
//							unsigned y = (in_base_y + my * 2 + j - shift.y + size_side) & (size_side - 1);
//							unsigned idx = encode(x, y);
//							unsigned val = get_val(h_input, idx);
//							local_mask |= (val & 1) << (j * 4 + i);
//						}
//					}
//					unsigned mid_val = mask8(local_mask);
//					h_mid[((ty * 2 + my) * mid_side + (tx * 2 + mx))] = mid_val;
//					combined |= (mid_val & 3) << ((my * 2 + mx) * 2);
//				}
//			}
//			h_top[tid] = mask8(combined);
//		}
//	}
//
//	dump2bit_grid(h_mid, "Mid");
//	dump2bit_grid(h_top, "Top");
//
//	// 3. Upload to GPU and launch kernel
//	uint64_t* d_input, * d_shift;
//	unsigned* d_mid, * d_top;
//	CHECK_CUDA(cudaMalloc(&d_input, total_words * sizeof(uint64_t)));
//	CHECK_CUDA(cudaMalloc(&d_shift, total_words * sizeof(uint64_t)));
//	CHECK_CUDA(cudaMalloc(&d_mid, mid_len * sizeof(unsigned)));
//	CHECK_CUDA(cudaMalloc(&d_top, top_len * sizeof(unsigned)));
//
//	CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), total_words * sizeof(uint64_t), cudaMemcpyHostToDevice));
//	setSZ0toConstantMem(size_side);
//	reduct << <(top_len + 255) / 256, 256 >> > (d_input, d_shift, d_mid, d_top, shift);
//	CHECK_CUDA(cudaDeviceSynchronize());
//
//	std::vector<unsigned> gpu_top(top_len);
//	CHECK_CUDA(cudaMemcpy(gpu_top.data(), d_top, top_len * sizeof(unsigned), cudaMemcpyDeviceToHost));
//
//	CHECK_CUDA(cudaFree(d_input));
//	CHECK_CUDA(cudaFree(d_shift));
//	CHECK_CUDA(cudaFree(d_mid));
//	CHECK_CUDA(cudaFree(d_top));
//
//	// 4. Compare
//	int errors = 0;
//	for(unsigned i = 0; i < top_len; ++i){
//		if(gpu_top[i] != h_top[i]){
//			printf("Mismatch at %u: expected %u, got %u\n", i, h_top[i], gpu_top[i]);
//			if(++errors > 10) break;
//		}
//	}
//	printf("Test completed with %d mismatches\n", errors);
//	return errors;
//}

int test_reduce(unsigned size_side = 32){
	srand(42);
	const unsigned total_values = size_side * size_side;
	const unsigned total_words = (total_values + 31) / 32;

	const unsigned mid_side = size_side >> 1;
	const unsigned mid_len = mid_side * mid_side;
	const unsigned total_words_mid = (mid_len + 15) / 16;

	const unsigned top_side = size_side >> 2;
	const unsigned top_len = top_side * top_side;
	const unsigned total_words_top = (mid_len + 15) / 16;

	std::vector<uint64_t> h_input(total_words, 0);
	std::vector<uint64_t> h_shifted(total_words, 0);
	std::vector<unsigned> h_mid(total_words_mid, 0);	// changed
	std::vector<unsigned> h_top(total_words_top, 0);	// changed

	// 1. Fill input with pseudo-random pattern
	for(unsigned i = 0; i < total_values; ++i){
		unsigned rnd1 = ((double)rand() / RAND_MAX) < 0.73 ? 0 : 2;
		unsigned rnd0 = ((double)rand() / RAND_MAX) < 0.77 ? 0 : 1;
		unsigned val = rnd1 + rnd0;
		unsigned word = i >> 5;
		unsigned bit = (i & 31) << 1;
		h_input[word] |= uint64_t(val & 0x3) << bit;
	}
	dump2bit_grid(h_input, "Input");
	// 2. Compute expected results on CPU
	auto get_val = [&](const std::vector<uint64_t>& d, unsigned idx) -> unsigned{
		return (d[idx >> 5] >> ((idx & 31) << 1)) & 0x3;
	};

	auto encode = [](unsigned x, unsigned y){
		return bit_reverse64(bit_reverse32(x) >> 1 | (bit_reverse32(y) >> 1) << 1) >> 32;
		//return __brevll(__brev(x) >> 1 | (__brev(y) >> 1) << 1) >> 32;
	};

	auto mask4 = [](unsigned a4){
		return (0b1111111011101000 >> (a4 & 15)) & 1;
	};
	auto mask8 = [&](unsigned a8){
		return mask4(a8 & 0xF) | (mask4((a8 >> 1) & 0xF) << 1);
	};

	int2 shift = {3, -5};

	for(unsigned ty = 0; ty < top_side; ++ty){
		for(unsigned tx = 0; tx < top_side; ++tx){
			unsigned in_base_x = tx * 4;
			unsigned in_base_y = ty * 4;
			unsigned tid = encode(tx, ty);

			// collect 4 mid-values
			unsigned combined = 0;
			for(unsigned my = 0; my < 2; ++my){
				for(unsigned mx = 0; mx < 2; ++mx){
					unsigned local_mask = 0;
					for(unsigned j = 0; j < 4; ++j){
						for(unsigned i = 0; i < 4; ++i){
							unsigned x = (in_base_x + mx * 2 + i - shift.x + size_side) & (size_side - 1);
							unsigned y = (in_base_y + my * 2 + j - shift.y + size_side) & (size_side - 1);
							unsigned idx = encode(x, y);
							unsigned val = get_val(h_input, idx);
							local_mask |= (val & 1) << (j * 4 + i);
						}
					}
					unsigned mid_val = mask8(local_mask);
					h_mid[((ty * 2 + my) * mid_side + (tx * 2 + mx))] = mid_val;
					combined |= (mid_val & 3) << ((my * 2 + mx) * 2);
				}
			}
			h_top[tid] = mask8(combined);
		}
	}
	//dump2bit_grid(h_input, "Shift");
	dump2bit_grid(h_mid, "Mid");
	dump2bit_grid(h_top, "Top");

	// 3. Upload to GPU and launch kernel
	uint64_t* d_input, * d_shift;
	unsigned* d_mid, * d_top;
	CHECK_CUDA(cudaMalloc(&d_input, total_words * sizeof(uint64_t)));
	CHECK_CUDA(cudaMalloc(&d_shift, total_words * sizeof(uint64_t)));
	CHECK_CUDA(cudaMalloc(&d_mid, mid_len * sizeof(unsigned)));
	CHECK_CUDA(cudaMalloc(&d_top, top_len * sizeof(unsigned)));

	CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), total_words * sizeof(uint64_t), cudaMemcpyHostToDevice));
	setSZ0toConstantMem(size_side);
	reduct << <(top_len + 255) / 256, 256 >> > (d_input, d_shift, d_mid, d_top, shift);
	CHECK_CUDA(cudaDeviceSynchronize());

	std::vector<unsigned> gpu_top(top_len);
	CHECK_CUDA(cudaMemcpy(gpu_top.data(), d_top, top_len * sizeof(unsigned), cudaMemcpyDeviceToHost));

	CHECK_CUDA(cudaFree(d_input));
	CHECK_CUDA(cudaFree(d_shift));
	CHECK_CUDA(cudaFree(d_mid));
	CHECK_CUDA(cudaFree(d_top));

	// 4. Compare
	int errors = 0;
	for(unsigned i = 0; i < top_len; ++i){
		if(gpu_top[i] != h_top[i]){
			printf("Mismatch at %u: expected %u, got %u\n", i, h_top[i], gpu_top[i]);
			if(++errors > 10) break;
		}
	}
	printf("Test completed with %d mismatches\n", errors);
	return errors;
}
int main(){
	for(int j = 1; j < 100; j++){
		if(!testreduct()){
			printf("Step %d. Error testreduct!", j);
			return -j;
		}
	}
	printf("\ntestreduct Ok\n");
	return 0;
	auto start = std::chrono::high_resolution_clock::now();
	int errors = test_reduce();
	auto end = std::chrono::high_resolution_clock::now();
	double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
	std::cout << "Total time: " << elapsed_ms << " ms. Errors=" << errors << "\n";
	return errors;
}