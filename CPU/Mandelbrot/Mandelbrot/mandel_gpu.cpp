#include <cstdlib>
#include <complex>
#include <fstream> 
#include <complex>
#include "pfc_types.h"
#include "pfc_bitmap_3.h"
#include <string_view>
#include <thread>
#include <future>
#include "mandelbrot.h"
#include "mandelbrotutil.h"
#include "kernel.cuh"
#include "pfc_threading.h"
#include <fstream>
#include <vector>
#include "mandel_cpu.h"

// define to store the bmp files
#define STOREIMAGES

int calculateOnDevice_opt_v1(int const current_iteration) {
	auto const memory_size{ PIXEL_PER_IMAGE * sizeof(pfc::byte_t) };

	// target host
	pfc::bitmap const bmp{ WIDTH, HEIGHT };
	//bmp.pixel_span().data()
	auto data{ bmp.pixel_span().data() };
	auto hp_destination{ std::make_unique<pfc::byte_t[]>(memory_size) };

	// target device
	pfc::byte_t* dp_destination{ nullptr };
	gpuErrchk(cudaMalloc(&dp_destination, memory_size));

	call_mandel_kernel3(BLOCKS, THREADS, dp_destination, PIXEL_PER_IMAGE, current_iteration);

	gpuErrchk(cudaPeekAtLastError());

	// retrieve result from GPU
	gpuErrchk(cudaMemcpy(hp_destination.get(), dp_destination, memory_size, cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(dp_destination));
	// copy to bmp
	for (auto i{ 0 }; i < PIXEL_PER_IMAGE; ++i) {
		data[i] = { 0, 0, hp_destination[i] };
	}
	// store test image
#ifdef STOREIMAGES
	bmp.to_file("../img/gpu_opt_v1" + std::to_string(current_iteration + 1) + ".bmp");
#endif // STOREIMAGES

	return 0;
}

int calculateOnDevice(int const current_iteration) {
	auto zoom_factor{ 1.0f };
	// calculate zoom_factor
	for (auto i{ 0 }; i < current_iteration; ++i) {
		zoom_factor *= ZOOM_FACTOR;
	}
	// set field of view
	auto const cx_min{ point_x + cx_min_factor * zoom_factor };
	auto const cx_max{ point_x + cx_max_factor * zoom_factor };
	auto const cy_min{ point_y + cy_min_factor * zoom_factor };
	auto const cy_max{ point_y + cy_max_factor * zoom_factor };

	auto const memory_size{ PIXEL_PER_IMAGE * sizeof(pfc::byte_t) };

	// target host
	pfc::bitmap const bmp{ WIDTH, HEIGHT };
	//bmp.pixel_span().data()
	auto data{ bmp.pixel_span().data() };
	auto hp_destination{ std::make_unique<pfc::byte_t[]>(memory_size) };

	// target device
	pfc::byte_t* dp_destination{ nullptr };
	gpuErrchk(cudaMalloc(&dp_destination, memory_size));

	call_mandel_kernel(BLOCKS, THREADS, dp_destination, PIXEL_PER_IMAGE, cx_min, cx_max, cy_min, cy_max);

	gpuErrchk(cudaPeekAtLastError());

	// retrieve result from GPU
	gpuErrchk(cudaMemcpy(hp_destination.get(), dp_destination, memory_size, cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(dp_destination));
	// copy to bmp
	for (auto i{ 0 }; i < PIXEL_PER_IMAGE; ++i) {
		data[i] = { 0, 0, hp_destination[i] };
	}
	// store test image
#ifdef STOREIMAGES
	bmp.to_file("../img/gpu_iter_" + std::to_string(current_iteration + 1) + ".bmp");
#endif // STOREIMAGES

	return 0;
}

int calculateOnDeviceBitmap(int const current_iteration) {
	auto zoom_factor{ 1.0f };
	// calculate zoom_factor
	for (auto i{ 0 }; i < current_iteration; ++i) {
		zoom_factor *= ZOOM_FACTOR;
	}
	// set field of view
	auto const cx_min{ point_x + cx_min_factor * zoom_factor };
	auto const cx_max{ point_x + cx_max_factor * zoom_factor };
	auto const cy_min{ point_y + cy_min_factor * zoom_factor };
	auto const cy_max{ point_y + cy_max_factor * zoom_factor };
	auto const memory_size{ PIXEL_PER_IMAGE * sizeof(pfc::BGR_4_t) };

	// target host
	pfc::bitmap const bmp{ WIDTH, HEIGHT };

	// target device
	pfc::BGR_4_t* dp_destination{ nullptr };
	gpuErrchk(cudaMalloc(&dp_destination, memory_size));

	call_mandel_kernel2(BLOCKS, THREADS, dp_destination, PIXEL_PER_IMAGE, cx_min, cx_max, cy_min, cy_max);

	gpuErrchk(cudaPeekAtLastError());

	// retrieve result from GPU
	gpuErrchk(cudaMemcpy(bmp.pixel_span().data(), dp_destination, memory_size, cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(dp_destination));
	// store test image
#ifdef STOREIMAGES
	bmp.to_file("../img/gpu_byte_" + std::to_string(current_iteration + 1) + ".bmp");
#endif // STOREIMAGES
	return 0;
}
// GPU functions
void sequential_gpu_byte(int const images) {
	// one thread per image
	for (auto o{ 0 }; o < images; ++o)
	{
		calculateOnDevice(o);
	}
}

void sequential_gpu_bitmap(int const images) {
	// one thread per image
	for (auto o{ 0 }; o < images; ++o)
	{
		calculateOnDeviceBitmap(o);
	}
}

void parallel_gpu_byte_all(int const images) {
	// one thread per image
	pfc::parallel_range(true, images, images, [](int o, int begin, int end) {
		calculateOnDevice(o);
		});
}

void parallel_gpu_byte_all_opt(int const images) {
	// one thread per image
	pfc::parallel_range(true, images, images, [](int o, int begin, int end) {
		calculateOnDevice_opt_v1(o);
		});
}

void parallel_gpu_bitmap_all(int const images) {
	// one thread per image
	pfc::parallel_range(true, images, images, [](int o, int begin, int end) {
		calculateOnDeviceBitmap(o);
		});
}

void parallel_gpu_bitmap_chunked(int const images, int const chunk_size) {
	// one thread per image
	pfc::parallel_range(true, chunk_size, images, [](int o, int begin, int end) {
		for (auto i{ begin }; i < end; ++i) {
			calculateOnDeviceBitmap(i);
		}
		});
}

void parallel_gpu_byte_chunked(int const images, int const chunk_size) {
	// one thread per image
	pfc::parallel_range(true, chunk_size, images, [](int o, int begin, int end) {
		for (auto i{ begin }; i < end; ++i)
			calculateOnDevice(i);
		});
}
