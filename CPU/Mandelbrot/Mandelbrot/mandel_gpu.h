#pragma once

#include "pfc_types.h"
#include <cuda_runtime.h>
#include "pfc_complex.h"
#include "mandel_constants_gpu.h"


// GPU functions
void sequential_gpu_byte(int const images);

void sequential_gpu_bitmap(int const images);

void parallel_gpu_byte_all(int const images);

void parallel_gpu_byte_all_opt(int const images);

void parallel_gpu_bitmap_all(int const images);

void parallel_gpu_bitmap_chunked(int const images, int const chunk_size);

void parallel_gpu_byte_chunked(int const images, int const chunk_size);


__device__ __forceinline__
pfc::byte_t valueDevice2(int const x, int const y, int const outer_index) {
	// calculate the constant
	pfc::complex<float> c(
		(CX_MIN[outer_index] + x / (WIDTH_FACTOR) * (CX_MAX[outer_index] - CX_MIN[outer_index])),
		(CY_MIN[outer_index] + y / (HEIGHT_FACTOR) * (CY_MAX[outer_index] - CY_MIN[outer_index]))
	);
	// initialize z
	pfc::complex<float> z(0.0f, 0.0f);
	auto iterations{ 0 };
	// calculate z
	while (z.norm() < 4 && iterations < ITERATIONS) {
		z = z.square() + c;
		++iterations;
	}
	// set color gradient
	return iterations < ITERATIONS ? COLORS[iterations] : 0;
}

__device__ __forceinline__
pfc::byte_t valueDevice(int const x, int const y, float const& cx_min, float const& cy_min, float const& cx_max, float const& cy_max, int const max_iterations) {
	// calculate the constant
	pfc::complex<float> c(
		(cx_min + x / (WIDTH - 1.0f) * (cx_max - cx_min)),
		(cy_min + y / (HEIGHT - 1.0f) * (cy_max - cy_min))
	);
	// initialize z
	pfc::complex<float> z(0.0f, 0.0f);
	auto iterations{ 0 };
	// calculate z

	while (z.norm() < 4 && iterations < max_iterations) {
		z = z.square() + c;
		++iterations;
	}
	// set color gradient
	return iterations < max_iterations ? 255 * iterations / ITERATIONS : 0;
}
