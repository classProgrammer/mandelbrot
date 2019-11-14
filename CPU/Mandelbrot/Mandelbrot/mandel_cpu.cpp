#include "mandel_cpu.h"
#include "mandel_constants_cpu.h"
#include "pfc_types.h"
#include "pfc_bitmap_3.h"
#include "pfc_threading.h"
#include <vector>

#include "pfc_complex.h"
// define to store the bmp files
//#define STOREIMAGES

// precalculated indices for value mapping
std::vector<int> X_VAL;
std::vector<int> Y_VAL;

pfc::byte_t valueHost2(int const inner_idx, int const outer_index) {
	// calculate the constant
	pfc::complex<float> c(
		(X_VAL[inner_idx] * X_FACTORS[outer_index] + CX_MIN[outer_index]),
		(Y_VAL[inner_idx] * Y_FACTORS[outer_index] + CY_MIN[outer_index])
	);
	// initialize z
	pfc::complex<float> z(0.0f, 0.0f);
	auto iterations{ 0 };
	// calculate z
	while (z.norm() < 4.0f && iterations++ < ITERATIONS) {
		z = z.square() + c;
	}
	// set color gradient
	return iterations < ITERATIONS ? COLORS[iterations] : 0;
}

pfc::byte_t valueHost(int const inner_idx, int const outer_index) {
	// calculate the constant
	float cr{ (X_VAL[inner_idx] * X_FACTORS[outer_index] + CX_MIN[outer_index]) }, 
		ci{ (Y_VAL[inner_idx] * Y_FACTORS[outer_index] + CY_MIN[outer_index]) };
	
	// initialize z

	int iterations{ ITERATIONS };
	float zr{ 0.0f }, 
		zi{ 0.0f }, 
		z_norm{ 0.0f }, 
		zr_2{ 0.0 }, 
		zi_2{0.0},
		tempi{ 0.0 };

	while (--iterations &&  z_norm < 4.0)
	{
		tempi = zr * zi;

		zi = tempi + tempi + ci;
		zr = zr_2 - zi_2 + cr;

		zr_2 = zr * zr;
		zi_2 = zi * zi;
		z_norm = zr_2 + zi_2;
	}
	iterations = ITERATIONS - iterations;
	// set color gradient
	return iterations < ITERATIONS ? COLORS[iterations] : 0;
}

// CPU functions
void global_sequential_local_sequential(int const images) {
	pfc::bitmap const bmp{ WIDTH, HEIGHT };
	auto data{ bmp.pixel_span().data() };

	for (auto o{ 0 }; o < images; ++o)
	{
		// foreach pixel in image
		for (auto i{ 0 }; i < PIXEL_PER_IMAGE; ++i) {
			data[i] = { 0, 0, valueHost(i, o) };
		}
#ifdef STOREIMAGES
		bmp.to_file("../img/cpu_gs_ls_" + std::to_string(o + 1) + ".bmp");
#endif // STOREIMAGES
	}
}

void global_parallel_local_sequential_task(int const images, int const outer_size) {
	// one thread per image
	pfc::parallel_range(true, outer_size, images, [](int outerIdx, int begin, int end) {
		pfc::bitmap const bmp{ WIDTH, HEIGHT };
		auto data{ bmp.pixel_span().data() };

		for (auto o{ begin }; o < end; ++o) {
			// foreach pixel in image
			for (auto i{ 0 }; i < PIXEL_PER_IMAGE; ++i) {
				data[i] = { 0, 0, valueHost(i, o) };
			}
#ifdef STOREIMAGES
			bmp.to_file("./cpu_gp_ls_" + std::to_string(o + 1) + ".bmp");
#endif // STOREIMAGES
		}
		});
}

void global_parallel_local_parallel_task(int const images, int const inner_size, int const outer_size) {
	// one thread per image
	pfc::parallel_range(true, outer_size, images, [inner_size](int thread_idx, int begin, int end) {

		pfc::bitmap const bmp{ WIDTH, HEIGHT };
		auto data{ bmp.pixel_span().data() };

		for (auto o{ begin }; o < end; ++o) {

			// foreach pixel in image
			pfc::parallel_range(true, inner_size, PIXEL_PER_IMAGE, [data, o](int innerIdx, int begin, int end) {
				for (auto i{ begin }; i < end; ++i) {
					data[i] = { 0, 0, valueHost(i, o) };
				}
				});
#ifdef STOREIMAGES
			bmp.to_file("../img/cpu_gp_lp_" + std::to_string(o + 1) + ".bmp");
#endif // STOREIMAGES
		}
		});
}

void global_parallel_local_parallel_task2(int const images, int const inner_size, int const outer_size) {
	// one thread per image
	pfc::parallel_range(true, outer_size, images, [inner_size](int thread_idx, int begin, int end) {

		pfc::bitmap const bmp{ WIDTH, HEIGHT };
		auto data{ bmp.pixel_span().data() };

		for (auto o{ begin }; o < end; ++o) {

			// foreach pixel in image
			pfc::parallel_range(true, inner_size, PIXEL_PER_IMAGE, [data, o](int innerIdx, int begin, int end) {
				for (auto i{ begin }; i < end; ++i) {
					data[i] = { 0, 0, valueHost2(i, o) };
				}
				});
#ifdef STOREIMAGES
			bmp.to_file("../img/cpu_gp_lp2_" + std::to_string(o + 1) + ".bmp");
#endif // STOREIMAGES
		}
		});
}

void global_sequential_local_prallel_task(int const images, int const inner_size) {
	// one thread per image
	pfc::bitmap const bmp{ WIDTH, HEIGHT };
	auto data{ bmp.pixel_span().data() };

	for (auto o{ 0 }; o < images; ++o)
	{
		// foreach pixel in image
		pfc::parallel_range(true, inner_size, PIXEL_PER_IMAGE, [data, o](int innerIdx, int begin, int end) {
			for (auto i{ begin }; i < end; ++i) {
				data[i] = { 0, 0, valueHost(i, o) };
			}
			});
#ifdef STOREIMAGES
		bmp.to_file("../img/cpu_gs_lp_" + std::to_string(o + 1) + ".bmp");
#endif // STOREIMAGES
	}
}

void global_parallel_local_sequential_thread(int const images, int const outer_size) {
	// one thread per image
	pfc::parallel_range(false, outer_size, images, [](int outerIdx, int begin, int end) {
		pfc::bitmap const bmp{ WIDTH, HEIGHT };
		auto data{ bmp.pixel_span().data() };
		
		for (auto o{ begin }; o < end; ++o) {
			// foreach pixel in image
			for (auto i{ 0 }; i < PIXEL_PER_IMAGE; ++i) {
				data[i] = { 0, 0, valueHost(i, o) };
			}
#ifdef STOREIMAGES
			bmp.to_file("../img/cpu_gp_ls_" + std::to_string(o + 1) + ".bmp");
#endif // STOREIMAGES
		}
		});
}

void global_parallel_local_parallel_thread(int const images, int const inner_size, int const outer_size) {
	// one thread per image
	pfc::parallel_range(false, outer_size, images, [inner_size](int thread_idx, int begin, int end) {

		pfc::bitmap const bmp{ WIDTH, HEIGHT };
		auto data{ bmp.pixel_span().data() };

		for (auto o{ begin }; o < end; ++o) {

			// foreach pixel in image
			pfc::parallel_range(false, inner_size, PIXEL_PER_IMAGE, [data, o](int innerIdx, int begin, int end) {
				for (auto i{ begin }; i < end; ++i) {
					data[i] = { 0, 0, valueHost(i, o) };
				}
				});
#ifdef STOREIMAGES
			bmp.to_file("../img/cpu_gp_lp_" + std::to_string(o + 1) + ".bmp");
#endif // STOREIMAGES
		}
		});
}

void global_sequential_local_prallel_thread(int const images, int const inner_size) {
	// one thread per image
	pfc::bitmap const bmp{ WIDTH, HEIGHT };
	auto data{ bmp.pixel_span().data() };
	
	for (auto o{ 0 }; o < images; ++o)
	{
		// foreach pixel in image
		pfc::parallel_range(false, inner_size, PIXEL_PER_IMAGE, [data, o](int innerIdx, int begin, int end) {
			for (auto i{ begin }; i < end; ++i) {
				data[i] = { 0, 0, valueHost(i, o) };
			}
			});
#ifdef STOREIMAGES
		bmp.to_file("../img/cpu_gs_lp_" + std::to_string(o + 1) + ".bmp");
#endif // STOREIMAGES
	}
}

void init_CPU() {
	for (auto i{ 0 }; i < PIXEL_PER_IMAGE; ++i) {
		X_VAL.emplace_back(i % WIDTH);
		Y_VAL.emplace_back(i / WIDTH);
	}
}