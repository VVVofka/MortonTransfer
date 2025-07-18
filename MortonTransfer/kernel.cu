#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <chrono>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// ===== ВСТАВЬ ЗДЕСЬ КОД ЯДРА transfer64_shared и утилиты get/set2bits64 из предыдущих сообщений =====
extern __shared__ uint64_t shared_buffer[]; // size = 2 * threads_per_block * sizeof(uint64_t)

static __device__ __forceinline__ uint64_t Part1By1(unsigned n){
	n &= 0x0000FFFF;
	n = (n | (n << 8)) & 0x00FF00FF;
	n = (n | (n << 4)) & 0x0F0F0F0F;
	n = (n | (n << 2)) & 0x33333333;
	n = (n | (n << 1)) & 0x55555555;
	return (uint64_t)n;
}

static __device__ __forceinline__ unsigned Compact1By1(uint64_t n){
	n &= 0x55555555ULL;
	n = (n | (n >> 1)) & 0x33333333ULL;
	n = (n | (n >> 2)) & 0x0F0F0F0FULL;
	n = (n | (n >> 4)) & 0x00FF00FFULL;
	n = (n | (n >> 8)) & 0x0000FFFFULL;
	return (unsigned)n;
}

static __device__ __forceinline__ uint64_t EncodeMorton2(unsigned x, unsigned y){
	return (Part1By1(y) << 1) | Part1By1(x);
}

static __device__ __forceinline__ unsigned DecodeMorton2X(uint64_t code){
	return Compact1By1(code);
}

static __device__ __forceinline__ unsigned DecodeMorton2Y(uint64_t code){
	return Compact1By1(code >> 1);
}

static __device__ __forceinline__ uint64_t MoveBase(uint64_t old_index, int shift_x, int shift_y, unsigned size){
	unsigned mask = size - 1;
	unsigned x = DecodeMorton2X(old_index);
	unsigned y = DecodeMorton2Y(old_index);
	unsigned new_x = (x + shift_x + size) & mask;
	unsigned new_y = (y + shift_y + size) & mask;
	return EncodeMorton2(new_x, new_y);
}
// ------------------------------------------------------------------------------------------
static __device__ __forceinline__ unsigned get2bits64(const uint64_t* data, unsigned index){
	unsigned word_index = index >> 5;              // index / 32
	unsigned bit_offset = (index & 31) << 1;       // (index % 32) * 2
	uint64_t word = data[word_index];
	return (word >> bit_offset) & 0x3;
}

static __device__ __forceinline__ void set2bits64(uint64_t* data, unsigned index, unsigned value){
	unsigned word_index = index >> 5;
	unsigned bit_offset = (index & 31) << 1;
	uint64_t mask = ~(0x3ULL << bit_offset);
	data[word_index] = (data[word_index] & mask) | ((uint64_t)(value & 0x3) << bit_offset);
}

static __device__ __forceinline__ unsigned get2bits64_local(uint64_t* data, unsigned index){
	unsigned word_index = index >> 5;
	unsigned bit_offset = (index & 31) << 1;
	uint64_t word = data[word_index];
	return (word >> bit_offset) & 0x3;
}

static __device__ __forceinline__ void set2bits64_local(uint64_t* data, unsigned index, unsigned value){
	unsigned word_index = index >> 5;
	unsigned bit_offset = (index & 31) << 1;
	uint64_t mask = ~(0x3ULL << bit_offset);
	auto comb = (data[word_index] & mask) | ((uint64_t)(value & 0x3) << bit_offset);
	data[word_index] = comb;	// not changed
	data[word_index] = 3;		// changed
}

static __global__ void transfer64_shared(
	const uint64_t* __restrict__ data_in,
	uint64_t* __restrict__ data_out,
	int2 shift,
	unsigned size_side){
	extern __shared__ uint64_t shared[]; // dynamically allocated: shared[0:in][n:out]
	uint64_t* buf_in = shared;
	uint64_t* buf_out = &shared[blockDim.x];

	unsigned total_values = size_side * size_side;
	unsigned global_index = blockIdx.x * blockDim.x + threadIdx.x;

	if(global_index >= total_values) return;

	// === Загрузка из глобальной памяти в shared memory ===
	unsigned value = get2bits64(data_in, global_index);
	buf_in[0] = 1;
	set2bits64_local(buf_in, threadIdx.x, value);

	__syncthreads();

	// === Перенос Morton-индекса внутри блока ===
	uint64_t new_index = MoveBase(global_index, shift.x, shift.y, size_side);
	const auto tmp = get2bits64_local(buf_in, threadIdx.x);
	set2bits64_local(buf_out, threadIdx.x, tmp);

	__syncthreads();

	// === Запись из shared memory в глобальную память ===
	unsigned final_value = get2bits64_local(buf_out, threadIdx.x);
	set2bits64(data_out, new_index, final_value);
}

// ---------------------------
// Хелперы для CPU (2-битные значения)
// ---------------------------
static void set2bits_host(std::vector<uint64_t>& buffer, unsigned index, unsigned value){
	unsigned word_index = index / 32;
	unsigned bit_offset = (index % 32) * 2;
	buffer[word_index] &= ~(0x3ULL << bit_offset);
	buffer[word_index] |= (uint64_t(value & 0x3) << bit_offset);
}

static unsigned get2bits_host(const std::vector<uint64_t>& buffer, unsigned index){
	unsigned word_index = index / 32;
	unsigned bit_offset = (index % 32) * 2;
	return (buffer[word_index] >> bit_offset) & 0x3;
}

static void dump2bit_grid(const std::vector<uint64_t>& data, unsigned size_side, const char* title = nullptr){
	if(title) std::cout << "--- " << title << " ---\n";
	for(unsigned y = 0; y < size_side; ++y){
		for(unsigned x = 0; x < size_side; ++x){
			unsigned idx = y * size_side + x;
			unsigned val = get2bits_host(data, idx);
			if(val)
				std::cout << val << " ";
			else
				std::cout << ". ";
		}
		std::cout << "\n";
	}
	std::cout << std::endl;
}
// ---------------------------
// MAIN
// ---------------------------
int main_(){
	const unsigned N = 5; // 2^5 = 32 x 32 2^12=4096
	const unsigned size_side = 1u << N;
	const unsigned total_values = size_side * size_side;
	const unsigned num_words = (total_values + 31) / 32;

	const size_t buffer_size = num_words * sizeof(uint64_t);
	const unsigned threads_per_block = 256;
	const size_t shared_mem_size = 2 * threads_per_block * sizeof(uint64_t);

	// Host buffers
	std::vector<uint64_t> h_input(num_words, 0);
	std::vector<uint64_t> h_result(num_words, 0);

	// Init test pattern for Test 1
	std::mt19937 rng(123);
	std::uniform_int_distribution<int> dist(0, 3);

	// --- Test 1: Set known values at edge and random positions ---
	std::cout << "[TEST 1] Boundary and random value check...\n";

	std::vector<unsigned> test_indices = {0, 1, total_values - 1};
	set2bits_host(h_input, test_indices[0], 3);
	set2bits_host(h_input, test_indices[1], 1);
	set2bits_host(h_input, test_indices[2], 2);
	dump2bit_grid(h_input, size_side, "Input");

	// Upload to device
	uint64_t* d_input;
	uint64_t* d_output;
	CHECK_CUDA(cudaMalloc(&d_input, buffer_size));
	CHECK_CUDA(cudaMalloc(&d_output, buffer_size));
	CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), buffer_size, cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemset(d_output, 0, buffer_size));

	dim3 blocks((total_values + threads_per_block - 1) / threads_per_block);
	int2 shift = {3, -2};

	transfer64_shared << <blocks, threads_per_block, shared_mem_size >> > (d_input, d_output, shift, size_side);
	CHECK_CUDA(cudaDeviceSynchronize());

	CHECK_CUDA(cudaMemcpy(h_result.data(), d_output, buffer_size, cudaMemcpyDeviceToHost));

	dump2bit_grid(h_result, size_side, "Result after shift");

	// Move values back on CPU and check
	for(unsigned idx : test_indices){
		unsigned value = get2bits_host(h_input, idx);
		// compute shifted index
		unsigned x = idx % size_side;
		unsigned y = idx / size_side;
		x = (x + shift.x + size_side) % size_side;
		y = (y + shift.y + size_side) % size_side;
		unsigned shifted_idx = (y * size_side) + x;

		unsigned res_val = get2bits_host(h_result, shifted_idx);
		printf("host=%u device=%u\n", res_val, value);
		assert(res_val == value);
	}

	std::cout << "Test 1 passed ✅\n";

	// --- Test 2: Repeated shifts with sum = 0 ---
	std::cout << "[TEST 2] Shift back-and-forth + timing...\n";

	// Initialize random values
	for(unsigned i = 0; i < total_values; ++i){
		set2bits_host(h_input, i, dist(rng));
	}

	CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), buffer_size, cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_output, d_input, buffer_size, cudaMemcpyDeviceToDevice)); // start from same

	const int2 shifts[] = {
		{5, -3}, {-2, 4}, {-3, -1}, {0, 0} // sum = 0
	};
	const int num_iters = sizeof(shifts) / sizeof(shifts[0]);

	auto start = std::chrono::high_resolution_clock::now();

	for(int i = 0; i < num_iters; ++i){
		transfer64_shared << <blocks, threads_per_block, shared_mem_size >> > (
			d_output, d_output, shifts[i], size_side);
	}

	CHECK_CUDA(cudaDeviceSynchronize());
	auto end = std::chrono::high_resolution_clock::now();

	double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
	std::cout << "Average time per iteration: " << (elapsed_ms / num_iters) << " ms\n";

	CHECK_CUDA(cudaMemcpy(h_result.data(), d_output, buffer_size, cudaMemcpyDeviceToHost));

	// Verify match
	for(unsigned i = 0; i < total_values; ++i){
		unsigned a = get2bits_host(h_input, i);
		unsigned b = get2bits_host(h_result, i);
		if(a != b){
			std::cerr << "Mismatch at index " << i << ": expected " << a << ", got " << b << "\n";
			std::exit(EXIT_FAILURE);
		}
	}

	std::cout << "Test 2 passed ✅\n";

	CHECK_CUDA(cudaFree(d_input));
	CHECK_CUDA(cudaFree(d_output));
	return 0;
}
