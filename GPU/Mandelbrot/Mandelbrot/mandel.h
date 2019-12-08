#pragma once
#include <cuda_runtime.h>
#include "pfc_types.h"
#include "mandel_constants_gpu.h"

// GPU functions
// sequential
void sequential_gpu(int const images);

// parallel
void parallel_streamed_GPU_prallel_range(int const images);
void parallel_streamed_GPU_for_loop(int const images);


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

