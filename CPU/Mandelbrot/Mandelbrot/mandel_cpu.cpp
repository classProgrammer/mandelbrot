#include "mandel_cpu.h"
#include "mandel_constants_cpu.h"
#include "pfc_types.h"
#include "pfc_bitmap_3.h"
#include "pfc_threading.h"
#include <vector>

#include "pfc_complex.h"

// precalculated indices for value mapping
std::vector<int> X_VAL;
std::vector<int> Y_VAL;

std::vector<pfc::bitmap> bitmaps;

pfc::byte_t valueHost(int const inner_idx, int const outer_index) {
	// calculate the constant
	float cr{ (X_VAL[inner_idx] * X_FACTORS[outer_index] + CX_MIN[outer_index]) }, 
		ci{ (Y_VAL[inner_idx] * Y_FACTORS[outer_index] + CY_MIN[outer_index]) };
	
	// initialize z

	auto iterations{ ITERATIONS };
	auto zr{ 0.0f }, 
		zi{ 0.0f }, 
		z_norm{ 0.0f }, 
		zr_2{ 0.0f }, 
		zi_2{0.0f},
		tempi{0.0f};

	while (--iterations &&  z_norm < BORDER)
	{
		tempi = zr * zi;
		zi =  tempi + tempi + ci;
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
	for (auto o{ 0 }; o < images; ++o)
	{
		auto data{bitmaps[o].pixel_span().data()};
		// foreach pixel in image
		for (auto i{ 0 }; i < PIXEL_PER_IMAGE; ++i) {
			data[i] = { 0, 0, valueHost(i, o) };
		}
	}
}

void global_parallel_local_sequential_task(int const images, int const outer_size) {
	// one thread per image
	pfc::parallel_range(true, outer_size, images, [](int outerIdx, int begin, int end) {
		for (auto o{ begin }; o < end; ++o) {
			auto data{ bitmaps[o].pixel_span().data() };
			// foreach pixel in image
			for (auto i{ 0 }; i < PIXEL_PER_IMAGE; ++i) {
				data[i] = { 0, 0, valueHost(i, o) };
			}
		}
		});
}

void global_parallel_local_parallel_task(int const images, int const inner_size, int const outer_size) {
	// one thread per image
	pfc::parallel_range(true, outer_size, images, [inner_size](int thread_idx, int begin, int end) {
		for (auto o{ begin }; o < end; ++o) {
			// foreach pixel in image
			pfc::parallel_range(true, inner_size, PIXEL_PER_IMAGE, [data{ bitmaps[o].pixel_span().data() }, o](int innerIdx, int begin, int end) {
				for (auto i{ begin }; i < end; ++i) {
					data[i] = { 0, 0, valueHost(i, o) };
				}
			});
		}
		});
}

void global_parallel_local_parallel_task2(int const images, int const inner_size) {
	// one thread per image
	pfc::parallel_range(true, images, images, [inner_size](int o, int begin, int end) {
		// foreach pixel in image
		pfc::parallel_range(true, inner_size, PIXEL_PER_IMAGE, [data{ bitmaps[o].pixel_span().data() }, o](int innerIdx, int begin, int end) {
			for (auto i{ begin }; i < end; ++i) {
				data[i] = { 0, 0, valueHost(i, o) };
			}
		});
		});
}

void global_sequential_local_prallel_task(int const images, int const inner_size) {
	for (auto o{ 0 }; o < images; ++o)
	{
		auto data{ bitmaps[o].pixel_span().data() };
		// foreach pixel in image
		pfc::parallel_range(true, inner_size, PIXEL_PER_IMAGE, [data, o](int innerIdx, int begin, int end) {
			for (auto i{ begin }; i < end; ++i) {
				data[i] = { 0, 0, valueHost(i, o) };
			}
			});
	}
}

void global_parallel_local_sequential_thread(int const images, int const outer_size) {
	// one thread per image
	pfc::parallel_range(false, outer_size, images, [](int outerIdx, int begin, int end) {		
		for (auto o{ begin }; o < end; ++o) {
			// foreach pixel in image
			auto data{ bitmaps[o].pixel_span().data() };
			for (auto i{ 0 }; i < PIXEL_PER_IMAGE; ++i) {
				data[i] = { 0, 0, valueHost(i, o) };
			}
		}
		});
}

void global_parallel_local_parallel_thread(int const images, int const inner_size, int const outer_size) {

	pfc::parallel_range(false, outer_size, images, [inner_size](int thread_idx, int begin, int end) {

		for (auto o{ begin }; o < end; ++o) {
			auto data{ bitmaps[o].pixel_span().data() };
			// foreach pixel in image
			pfc::parallel_range(false, inner_size, PIXEL_PER_IMAGE, [data, o](int innerIdx, int begin, int end) {
				for (auto i{ begin }; i < end; ++i) {
					data[i] = { 0, 0, valueHost(i, o) };
				}
				});
		}
		});
}

void global_sequential_local_prallel_thread(int const images, int const inner_size) {
	for (auto o{ 0 }; o < images; ++o)
	{
		auto data{ bitmaps[o].pixel_span().data() };
		// foreach pixel in image
		pfc::parallel_range(false, inner_size, PIXEL_PER_IMAGE, [data, o](int innerIdx, int begin, int end) {
			for (auto i{ begin }; i < end; ++i) {
				data[i] = { 0, 0, valueHost(i, o) };
			}
			});
	}
}

void init_CPU() {

	static const int max_images{ 200 };

	for (auto i{ 0 }; i < PIXEL_PER_IMAGE; ++i) {
		X_VAL.emplace_back(i % WIDTH);
		Y_VAL.emplace_back(i / WIDTH);
	}

	for (auto i{ 0 }; i < max_images; ++i) {
		bitmaps.emplace_back(pfc::bitmap{ WIDTH, HEIGHT });
	}
}

void storeImagesCPU(int const images, std::string const &prefix) {
	for (auto o{ 0 }; o < images; ++o) {
		bitmaps[o].to_file("../img/" + prefix + "_" + std::to_string(o + 1) + ".bmp");
	}
}