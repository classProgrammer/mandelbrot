#pragma once
#include <cuda_runtime.h>
#include<iostream>
#include "pfc_types.h"
#include <complex>
#include <cuComplex.h>
#include "pfc_complex.h"
#include "mandel_constants.h"


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
	return iterations < max_iterations ? 255.0 * iterations / ITERATIONS : 0;
}