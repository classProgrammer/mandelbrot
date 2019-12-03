#pragma once
#include <cuda_runtime.h>
#include "pfc_types.h"
#include "mandel_constants_gpu.h"

// CPU functions
// GS LS
void global_sequential_local_sequential(int const images);

// GP LS
void global_parallel_local_sequential_thread(int const images);
void global_parallel_local_sequential_task(int const images);

// GS LP
void global_sequential_local_prallel_thread(int const images, int const inner_size);
void global_sequential_local_prallel_task(int const images, int const inner_size);

// GP LP
void global_parallel_local_parallel_thread(int const images, int const inner_size);
void global_parallel_local_parallel_task(int const images, int const inner_size);
//void global_parallel_local_parallel_task3(int const images, int const inner_size);


// GPU functions
// sequential
void sequential_gpu(int const images);

// parallel
void parallel_streamed_GPU_prallel_range(int const images);
void parallel_streamed_GPU_for_loop(int const images);
void parallel_GPU_stream0(int const images);

// Lifecycle methods
void initCPU();
void initGPU();
void freeGPU();

// Shared Methods
void storeLastImage(int const calculated_images, std::string const& prefix);
void storeImages(int const calculated_images, std::string const& prefix);

__device__ __forceinline__
pfc::byte_t colorDevice(float const cr, float const ci) {
	auto iterations{ START_ITERATION };
	auto zr{ 0.0f },
		zi{ 0.0f },
		zr_sq{ 0.0f },
		zi_sq{ 0.0f },
		temp_i{ 0.0f };

	do {
		zr_sq = zr * zr;
		zi_sq = zi * zi;
		temp_i = zr * zi;
		zi = temp_i + temp_i + ci;
		zr = zr_sq - zi_sq + cr;

	} while (zr_sq + zi_sq < BORDER && --iterations);

	return COLORS[iterations];
}

