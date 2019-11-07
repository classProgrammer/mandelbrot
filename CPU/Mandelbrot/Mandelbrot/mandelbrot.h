#pragma once
#include <cuda_runtime.h>
#include<iostream>
#include "pfc_types.h"
#include <complex>
#include <cuComplex.h>
#include "pfc_complex.h"

// Point to zoom at
auto constexpr point_x{ -0.745289981f };
auto constexpr point_y{ 0.113075003f };

// format 2:3
__device__
auto const WIDTH{ 8192 };
//auto const WIDTH{ 819 };
__device__
auto const HEIGHT{ 4608 };
//auto const HEIGHT{ 460 };

// Zoom factor per image
constexpr auto ZOOM_FACTOR{ 0.95f };
// params
__device__
auto const ITERATIONS{ 50 };
// from
auto constexpr cx_min_factor{ -2.0f };
auto constexpr cy_min_factor{ -1.125f };
// to
auto constexpr cx_max_factor{ 2.0f };
auto constexpr cy_max_factor{ 1.125f };
// color upper bound
auto constexpr MAX_COLOR{ 255 };
// graphics card props and calculation
auto constexpr THREADS{ 1024 };
auto constexpr BLOCKS{ 36864 };
auto constexpr COLOR_FACTOR{ (float)MAX_COLOR / ITERATIONS };

auto constexpr AVAILABLE_POWER{ BLOCKS * THREADS }; // 20 * 1024 = 20480
auto constexpr PIXEL_PER_IMAGE{ WIDTH * HEIGHT }; // 37.748.736
//auto constexpr PIXEL_PER_THREAD{ (float)PIXEL_PER_IMAGE / AVAILABLE_POWER };
auto constexpr PIXEL_PER_THREAD{ 2048.0f }; // to make us of a single byte array efficiently the pixels were adjusted to 2048 per thrad instead of 1843,2 pixels
// 2048 pixels means no line break logic is needed in the calculation to calculate the 2D index of the 1D array
__device__
auto const WIDTH_FACTOR{ WIDTH - 1.0f };
__device__
auto const HEIGHT_FACTOR{ HEIGHT - 1.0f };
__device__
pfc::byte_t const COLORS[] = { 0 ,5 ,10 ,15 ,20 ,25 ,30 ,35 ,40 ,45 ,51 ,56 ,61 ,66 ,71 ,76 ,81 ,86 ,91 ,96 ,102 ,107 ,112 ,117 ,122 ,127 ,132 ,137 ,142 ,147 ,153 ,158 ,163 ,168 ,173 ,178 ,183 ,188 ,193 ,198 ,204 ,209 ,214 ,219 ,224 ,229 ,234 ,239 ,244 ,249 };

__host__ __forceinline__
pfc::byte_t valueHost(int const x, int const y, float const& cx_min, float const& cy_min, float const& cx_max, float const& cy_max, int const max_iterations) {
	// calculate the constant
	pfc::complex<float> c(
		(cx_min + x / (WIDTH_FACTOR) * (cx_max - cx_min)),
		(cy_min + y / (HEIGHT_FACTOR) * (cy_max - cy_min))
	);
	// initialize z
	pfc::complex<float> z(0.0f, 0.0f);
	auto iterations{ 0 };
	// calculate z
	while (z.norm() < 4 && iterations < max_iterations) {
		z = z * z + c;
		++iterations;
	}
	// set color gradient
	return iterations < max_iterations ? COLORS[iterations] : 0;
}


__device__ __forceinline__
pfc::byte_t valueDevice(int const x, int const y, float const& cx_min, float const& cy_min, float const& cx_max, float const& cy_max, int const max_iterations) {
	// calculate the constant
	pfc::complex<float> c(
		(cx_min + x / (WIDTH_FACTOR) * (cx_max - cx_min)),
		(cy_min + y / (HEIGHT_FACTOR) * (cy_max - cy_min))
	);
	// initialize z
	pfc::complex<float> z(0.0f, 0.0f);
	auto iterations{ 0 };
	// calculate z

	while (z.norm() < 4 && iterations < max_iterations) {
		z = z * z + c;
		iterations++;
	}
	// set color gradient
	return iterations < max_iterations ? COLORS[iterations] : 0;
}