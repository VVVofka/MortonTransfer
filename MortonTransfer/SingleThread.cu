#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <chrono>
#include <string>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

static __device__ __host__ __forceinline__ unsigned DecodeMorton2X(unsigned code){
	code &= 0x55555555;
	code = (code ^ (code >> 1)) & 0x33333333;
	code = (code ^ (code >> 2)) & 0x0F0F0F0F;
	code = (code ^ (code >> 4)) & 0x00FF00FF;
	code = (code ^ (code >> 8)) & 0x0000FFFF;
	return code;
}

static __device__ __host__ __forceinline__ unsigned DecodeMorton2Y(unsigned code){
	code >>= 1;
	code &= 0x55555555;
	code = (code ^ (code >> 1)) & 0x33333333;
	code = (code ^ (code >> 2)) & 0x0F0F0F0F;
	code = (code ^ (code >> 4)) & 0x00FF00FF;
	code = (code ^ (code >> 8)) & 0x0000FFFF;
	return code;
}

static __device__ __host__ __forceinline__ unsigned EncodeMorton2(unsigned x, unsigned y){
	x &= 0x0000ffff;
	y &= 0x0000ffff;
	x = (x | (x << 8)) & 0x00FF00FF;
	y = (y | (y << 8)) & 0x00FF00FF;
	x = (x | (x << 4)) & 0x0F0F0F0F;
	y = (y | (y << 4)) & 0x0F0F0F0F;
	x = (x | (x << 2)) & 0x33333333;
	y = (y | (y << 2)) & 0x33333333;
	x = (x | (x << 1)) & 0x55555555;
	y = (y | (y << 1)) & 0x55555555;
	return x | (y << 1);
}

static __device__ __host__ __forceinline__ unsigned get2bits64(const uint64_t* data, unsigned index){
	unsigned word_index = index >> 5;
	unsigned bit_offset = (index & 31) << 1;
	return (data[word_index] >> bit_offset) & 0x3;
}

static __device__ __host__ __forceinline__ void set2bits64(uint64_t* data, unsigned index, unsigned value){
	unsigned word_index = index >> 5;
	unsigned bit_offset = (index & 31) << 1;
	uint64_t mask = ~(0x3ULL << bit_offset);
	data[word_index] = (data[word_index] & mask) | ((uint64_t)(value & 0x3) << bit_offset);
}

static void dump2bit_grid(const std::vector<uint64_t>& data, unsigned size_side, const char* title = nullptr){
	if(title) std::cout << "--- " << title << " ---\n";
	for(unsigned y = 0; y < size_side; ++y){
		for(unsigned x = 0; x < size_side; ++x){
			unsigned idx_morton = EncodeMorton2(x, y);
			unsigned val = get2bits64(data.data(), idx_morton);
			if(val)
				std::cout << val << " ";
			else
				std::cout << ". ";
		}
		std::cout << "\n";
	}
	std::cout << std::endl;
}
// -------------------------------------------------------------------------
static __global__ void transfer64_tile(const uint64_t* __restrict__ data_in,
								uint64_t* __restrict__ data_out,
								int2 shift,
								unsigned size_side){
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t result = 0;
#pragma unroll
	for(unsigned i = 0; i < 32; ++i){
		unsigned out_index = (tid << 5) + i;

		// Координаты в новом базисе (где мы записываем)
		unsigned x = DecodeMorton2X(out_index);
		unsigned y = DecodeMorton2Y(out_index);

		// Перенос координат в старый базис (откуда читаем)
		unsigned x_in = (x - shift.x + size_side) & (size_side - 1);
		unsigned y_in = (y - shift.y + size_side) & (size_side - 1);

		unsigned in_index = EncodeMorton2(x_in, y_in);
		unsigned val = get2bits64(data_in, in_index);

		result |= uint64_t(val & 0x3) << (i * 2);
	}
	data_out[tid] = result;
}
// ---------------------------
// MAIN
// ---------------------------
void test1(){
	const unsigned N = 5; // 2^5 = 32 x 32 2^12=4096
	const unsigned size_side = 1u << N;
	const unsigned total_values = size_side * size_side;
	const unsigned total_words = (total_values + 31) / 32;

	const size_t buffer_size = total_words * sizeof(uint64_t);
	const unsigned threads_per_block = 128;

	// Host buffers
	std::vector<uint64_t> h_input(total_words, 0);
	std::vector<uint64_t> h_result(total_words, 0);

	for(unsigned y = 0; y < size_side; y++){
		for(unsigned x = 0; x < size_side; x++){
			unsigned id_morton = EncodeMorton2(x, y);
			unsigned newy = (y < size_side / 2) ? 2 : 0;
			unsigned newx = (x < size_side / 2) ? 1 : 0;
			unsigned val = newx + newy;
			set2bits64(h_input.data(), id_morton, val);	//25:(5,2)
		}
	}

	// --- Test 1: Set known values at edge and random positions ---
	std::cout << "[TEST 1] Boundary and random value check...\n";
	unsigned write_val = 3; // 2 bit
	set2bits64(h_input.data(), 1022, write_val);	//25:(5,2)
	dump2bit_grid(h_input, size_side, "Input");

	// Upload to device
	uint64_t* d_input, * d_output;
	CHECK_CUDA(cudaMalloc(&d_input, buffer_size));
	CHECK_CUDA(cudaMalloc(&d_output, buffer_size));
	CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), buffer_size, cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemset(d_output, 0, buffer_size));
	unsigned blocks = (total_words + threads_per_block - 1) / threads_per_block;
	int2 shift = {-3, -1}; // to 2 1 (6)

	transfer64_tile << <blocks, threads_per_block >> > (d_input, d_output, shift, size_side);
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaMemcpy(h_result.data(), d_output, buffer_size, cudaMemcpyDeviceToHost));
	std::string s = "Result after shift " + std::to_string(shift.x) + ", " + std::to_string(shift.y);
	dump2bit_grid(h_result, size_side, s.c_str());

	// Move values back on CPU and check
	unsigned res_val = get2bits64(h_result.data(), 6);
	if(res_val == write_val)
		std::cout << "Test 1 passed\n";
	else
		std::cout << "Test 1 failed write:" << write_val << " read:" << res_val << "\n";
}

double test2(unsigned N = 10, int num_iters = 1000, unsigned threads_per_block = 256){
	const unsigned size_side = 1u << N;
	std::cout << "TEST2 iterations=" << num_iters << " Side=" << size_side <<
		" Threads per block=" << threads_per_block << "...\n";
	const uint64_t total_values = (uint64_t)size_side * (uint64_t)size_side;
	const uint64_t total_words = (total_values + 31) / 32;
	const unsigned blocks = unsigned((total_words + threads_per_block - 1) / threads_per_block);

	// Host buffers
	std::vector<uint64_t> h_input(total_words, 0);
	std::vector<uint64_t> h_result(total_words, 0);

	// Upload to device
	const size_t buffer_size = total_words * sizeof(uint64_t);
	uint64_t* d_input, * d_output;
	CHECK_CUDA(cudaMalloc(&d_input, buffer_size));
	CHECK_CUDA(cudaMalloc(&d_output, buffer_size));

	// Initialize random values
	std::mt19937 rng(123);
	std::uniform_int_distribution<int> dist(0, 3);
	for(unsigned i = 0; i < total_values; ++i)
		set2bits64(h_input.data(), i, dist(rng));

	CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), buffer_size, cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_output, d_input, buffer_size, cudaMemcpyDeviceToDevice)); // start from same

	std::vector<int2> shifts(num_iters);
	int2 sum{0, 0};
	std::uniform_int_distribution<int> dist_shifts(1 - (int)size_side, size_side - 1);
	for(int j = 1; j < num_iters; j++){
		shifts[j].x = dist_shifts(rng);
		sum.x = (sum.x + shifts[j].x + size_side) % size_side;
		shifts[j].y = dist_shifts(rng);
		sum.y = (sum.y + shifts[j].y + size_side) % size_side;
	}
	shifts[0] = {-sum.x, -sum.y};
	auto start = std::chrono::high_resolution_clock::now();
	for(int i = 0; i < num_iters; ++i){
		transfer64_tile << <blocks, threads_per_block >> >(d_input, d_output, shifts[i], size_side);
		CHECK_CUDA(cudaDeviceSynchronize());
		CHECK_CUDA(cudaMemcpy(d_input, d_output, buffer_size, cudaMemcpyDeviceToDevice));
	}
	auto end = std::chrono::high_resolution_clock::now();
	double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
	double avd_time = elapsed_ms / num_iters;
	std::cout << "Average time per iteration: " << avd_time << " ms.";

	CHECK_CUDA(cudaMemcpy(h_result.data(), d_output, buffer_size, cudaMemcpyDeviceToHost));

	// Verify match
	for(unsigned i = 0; i < total_values; ++i){
		unsigned a = get2bits64(h_input.data(), i);
		unsigned b = get2bits64(h_result.data(), i);
		if(a != b){
			std::cerr << "\nMismatch at index " << i << ": expected " << a << ", got " << b << "\n";
			return 0;	//			std::exit(EXIT_FAILURE);
		}
	}
	std::cout << " Test2 OK\n";
	CHECK_CUDA(cudaFree(d_input));
	CHECK_CUDA(cudaFree(d_output));
	return avd_time;
}
// ---------------------------
int main(){
	auto start = std::chrono::high_resolution_clock::now();
	int errors = 0;
	//test1(); return errors;
//#define SINGLE
#ifdef SINGLE
	double minval = DBL_MAX;;
	for(int n = 0; n < 8; n++){
		double avg_time = test2(12, 1000, 256);
		if(avg_time == 0)
			return -1;
		if(avg_time < minval)
			minval = avg_time;
	}
	printf("minval=%f \n", minval);
	return 0;
#else
	int num_iters = 1000;
	// size_side = 2^N
	for(unsigned N = 6; N <= 12; N++){ // max <= 15 (32'768)
		printf("N = %u\n", N);
		for(unsigned threads_per_block = 16; threads_per_block <= 1024; threads_per_block *= 2){
			double avg_time = test2(N, num_iters, threads_per_block);
			if(avg_time == 0)
				errors++;
			if(errors) return errors;
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
	std::cout << "Total time: " << elapsed_ms << " ms.\n";
	return errors;
#endif // SINGLE
}