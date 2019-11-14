#include "kernel.cuh"
#include "mandel_gpu.h"
#include "mandel_constants_gpu.h"

// grid_dim => number of blocks in the grid
// a grid can have x blocks 'cores'
// a block can have x threads (2048)

//pfc::byte_t value(int const x, int const y, float const& cx_min, float const& cy_min, float const& cx_max, float const& cy_max, int const max_iterations)

// kernel functions => __global__
__global__ void mandel_kernel(pfc::byte_t* dest, int size,
	float const cx_min, float const cx_max, float const cy_min, float const cy_max) {
	// threads are always generated in groups of 32 => pay attention on idx

	// all threads are here at the same time => parallel
	//auto t{ blockIdx.x * blockDim.x + threadIdx.x }; // absolute thread idx

	auto const thread_index{(blockIdx.x * blockDim.x + threadIdx.x)};// start at position x

	// copy at thread position if the position is valid
	if (thread_index < size) {
		auto x{ thread_index % WIDTH };
		auto y{ thread_index / WIDTH };
		dest[thread_index] = valueDevice(x, y, cx_min, cy_min, cx_max, cy_max, ITERATIONS);
	}
}

__global__ void mandel_kernel2(pfc::BGR_4_t* dest, int size,
	float const cx_min, float const cx_max, float const cy_min, float const cy_max) {
	// threads are always generated in groups of 32 => pay attention on idx

	// all threads are here at the same time => parallel
	//auto t{ blockIdx.x * blockDim.x + threadIdx.x }; // absolute thread idx

	auto const thread_index{ (blockIdx.x * blockDim.x + threadIdx.x) };// start at position x

	// copy at thread position if the position is valid
	if (thread_index < size) {
		auto x{ thread_index % WIDTH };
		auto y{ thread_index / WIDTH };
		dest[thread_index] = { 0, 0, valueDevice(x, y, cx_min, cy_min, cx_max, cy_max, ITERATIONS) };
	}
}

__global__ void mandel_kernel3(pfc::byte_t* dest, int size, int const outer_idx) {
	// threads are always generated in groups of 32 => pay attention on idx

	// all threads are here at the same time => parallel
	//auto t{ blockIdx.x * blockDim.x + threadIdx.x }; // absolute thread idx

	auto const thread_index{ (blockIdx.x * blockDim.x + threadIdx.x) };// start at position x

	// copy at thread position if the position is valid
	if (thread_index < size) {
		auto x{ thread_index % WIDTH };
		auto y{ thread_index / WIDTH };
		dest[thread_index] = valueDevice3(x, y, outer_idx);
	}
}

// delegeate params from host -> device
void call_mandel_kernel(dim3 const& big, dim3 const& tib, pfc::byte_t* dest, int const size,
	float const& cx_min, float const& cx_max, float const& cy_min, float const& cy_max) {
	// <<<>>> = kernel configuration, run with big many blocks with tib many threads in each block
	mandel_kernel <<<big, tib >>> (dest, size, cx_min, cx_max, cy_min, cy_max);
}

void call_mandel_kernel2(dim3 const& big, dim3 const& tib, pfc::BGR_4_t* dest, int const size,
	float const& cx_min, float const& cx_max, float const& cy_min, float const& cy_max) {
	// <<<>>> = kernel configuration, run with big many blocks with tib many threads in each block
	mandel_kernel2 << <big, tib >> > (dest, size, cx_min, cx_max, cy_min, cy_max);
}

void call_mandel_kernel3(dim3 const& big, dim3 const& tib, pfc::byte_t* dest, int const size, int const outer_idx) {
	// <<<>>> = kernel configuration, run with big many blocks with tib many threads in each block
	mandel_kernel3 << <big, tib >> > (dest, size, outer_idx);
}
