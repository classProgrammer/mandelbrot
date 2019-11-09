//#include "mandel_cpu.h"
//#include "mandel_constants.h"
//#include <cstdlib>
//#include <complex>
//#include <fstream>
//#include <complex>
//#include "pfc_types.h"
//#include "pfc_bitmap_3.h"
//#include <string_view>
//#include <thread>
//#include <future>
//#include "mandelbrot.h"
//#include "kernel.cuh"
//#include "pfc_threading.h"
//#include <fstream>
//#include <vector>
//
//// precalculated indices for value mapping
//std::vector<int> X_VAL;
//std::vector<int> Y_VAL;
//
//pfc::byte_t valueHost_opt(int const inner_idx, int const outer_index) {
//	// calculate the constant
//	pfc::complex<float> c(
//		(CX_MIN[outer_index] + X_VAL[inner_idx] / WIDTH_FACTOR * (CX_MAX[outer_index] - CX_MIN[outer_index])),
//		(CY_MIN[outer_index] + Y_VAL[inner_idx] / HEIGHT_FACTOR * (CY_MAX[outer_index] - CY_MIN[outer_index]))
//	);
//	// initialize z
//	pfc::complex<float> z(0.0f, 0.0f);
//	auto iterations{ 0 };
//	// calculate z
//	while (z.norm() < 4 && iterations++ < ITERATIONS) {
//		z = z.square() + c;
//	}
//	// set color gradient
//	return iterations < ITERATIONS ? COLORS[iterations] : 0;
//}
//
//pfc::byte_t valueHost_opt_v2(int const inner_idx, int const outer_index) {
//	// calculate the constant
//	pfc::complex<float> c(
//		(X_VAL[inner_idx] * X_FACTORS[outer_index] + CX_MIN[outer_index]),
//		(Y_VAL[inner_idx] * Y_FACTORS[outer_index] + CY_MIN[outer_index])
//	);
//	// initialize z
//	pfc::complex<float> z(0.0f, 0.0f);
//	auto iterations{ 0 };
//	// calculate z
//	while (z.norm() < 4 && iterations++ < ITERATIONS) {
//		z = z.square() + c;
//	}
//	// set color gradient
//	return iterations < ITERATIONS ? COLORS[iterations] : 0;
//}
//
//// CPU functions
//void global_sequential_local_sequential(int const images) {
//
//	for (auto o{ 0 }; o < images; ++o)
//	{
//		pfc::bitmap const bmp{ WIDTH, HEIGHT };
//		auto data{ bmp.pixel_span().data() };
//
//		// foreach pixel in image
//		for (auto i{ 0 }; i < PIXEL_PER_IMAGE; ++i) {
//			data[i] = { 0, 0, valueHost(X_VAL[i], Y_VAL[i], CX_MIN[o], CY_MIN[o], CX_MAX[o], CY_MAX[o], ITERATIONS) };
//		}
//#ifdef STOREIMAGES
//		bmp.to_file("../img/cpu_gs_ls" + std::to_string(o + 1) + ".bmp");
//#endif // STOREIMAGES
//	}
//}
//
//void global_parallel_local_sequential(int const images, int const outer_size) {
//	// one thread per image
//	pfc::parallel_range(true, outer_size, images, [](int outerIdx, int begin, int end) {
//		pfc::bitmap const bmp{ WIDTH, HEIGHT };
//		auto data{ bmp.pixel_span().data() };
//
//		// foreach pixel in image
//		for (auto i{ 0 }; i < PIXEL_PER_IMAGE; ++i) {
//			data[i] = { 0, 0, valueHost(X_VAL[i], Y_VAL[i], CX_MIN[outerIdx], CY_MIN[outerIdx], CX_MAX[outerIdx], CY_MAX[outerIdx], ITERATIONS) };
//		}
//#ifdef STOREIMAGES
//		bmp.to_file("../img/cpu_gp_ls" + std::to_string(outerIdx + 1) + ".bmp");
//#endif // STOREIMAGES
//		});
//}
//
//void global_parallel_local_parallel(int const images, int const inner_size, int const outer_size) {
//	// one thread per image
//	pfc::parallel_range(true, outer_size, images, [inner_size](int thread_idx, int begin, int end) {
//		// o = outer
//		// i = inner
//		for (auto o{ begin }; o < end; ++o) {
//			pfc::bitmap const bmp{ WIDTH, HEIGHT };
//			auto data{ bmp.pixel_span().data() };
//
//			// foreach pixel in image
//			pfc::parallel_range(true, inner_size, PIXEL_PER_IMAGE, [data, o](int innerIdx, int begin, int end) {
//				for (auto i{ begin }; i < end; ++i) {
//					data[i] = { 0, 0, valueHost(X_VAL[i], Y_VAL[i], CX_MIN[o], CY_MIN[o], CX_MAX[o], CY_MAX[o], ITERATIONS) };
//				}
//				});
//#ifdef STOREIMAGES
//			bmp.to_file("../img/cpu_gp_lp" + std::to_string(o + 1) + ".bmp");
//#endif // STOREIMAGES
//		}
//		});
//}
//
//void global_parallel_local_parallel_v2(int const images, int const inner_size, int const outer_size) {
//	// one thread per image
//	pfc::parallel_range(true, outer_size, images, [inner_size](int thread_idx, int begin, int end) {
//		// o = outer
//		// i = inner
//		for (auto o{ begin }; o < end; ++o) {
//			pfc::bitmap const bmp{ WIDTH, HEIGHT };
//			auto data{ bmp.pixel_span().data() };
//
//			// foreach pixel in image
//			pfc::parallel_range(true, inner_size, PIXEL_PER_IMAGE, [data, o](int innerIdx, int begin, int end) {
//				for (auto i{ begin }; i < end; ++i) {
//					data[i] = { 0, 0, valueHost_opt(i, o) };
//				}
//				});
//#ifdef STOREIMAGES
//			bmp.to_file("../img/cpu_gp_lp_v2" + std::to_string(o + 1) + ".bmp");
//#endif // STOREIMAGES
//		}
//		});
//}
//
//
//void global_parallel_local_parallel_v3(int const images, int const inner_size, int const outer_size) {
//	// one thread per image
//	pfc::parallel_range(true, outer_size, images, [inner_size](int thread_idx, int begin, int end) {
//		// o = outer
//		// i = inner
//		for (auto o{ begin }; o < end; ++o) {
//			pfc::bitmap const bmp{ WIDTH, HEIGHT };
//			auto data{ bmp.pixel_span().data() };
//
//			// foreach pixel in image
//			pfc::parallel_range(true, inner_size, PIXEL_PER_IMAGE, [data, o](int innerIdx, int begin, int end) {
//				for (auto i{ begin }; i < end; ++i) {
//					data[i] = { 0, 0, valueHost_opt_v2(i, o) };
//				}
//				});
//#ifdef STOREIMAGES
//			bmp.to_file("../img/cpu_gp_lp_v3" + std::to_string(o + 1) + ".bmp");
//#endif // STOREIMAGES
//		}
//		});
//}
//
//void global_sequential_local_prallel(int const images, int const inner_size) {
//	// one thread per image
//	for (auto o{ 0 }; o < images; ++o)
//	{
//		pfc::bitmap const bmp{ WIDTH, HEIGHT };
//		auto data{ bmp.pixel_span().data() };
//
//		// foreach pixel in image
//		pfc::parallel_range(true, inner_size, PIXEL_PER_IMAGE, [data, o](int innerIdx, int begin, int end) {
//			for (auto i{ begin }; i < end; ++i) {
//				data[i] = { 0, 0, valueHost(X_VAL[i], Y_VAL[i], CX_MIN[o], CY_MIN[o], CX_MAX[o], CY_MAX[o], ITERATIONS) };
//			}
//			});
//#ifdef STOREIMAGES
//		bmp.to_file("../img/cpu_gs_lp" + std::to_string(o + 1) + ".bmp");
//#endif // STOREIMAGES
//	}
//}
//
//void init() {
//	for (auto i{ 0 }; i < PIXEL_PER_IMAGE; ++i) {
//		auto x{ i % WIDTH };
//		auto y{ i / WIDTH };
//
//		X_VAL.emplace_back(x);
//		Y_VAL.emplace_back(y);
//	}
//}