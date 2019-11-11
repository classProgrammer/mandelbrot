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

// TODO:
// CPU threads vs task with param settings when is what better
// try to be creative with paralelism on CPU, e.g. 4 parallel images which are calculated parallel
// c++ complex class => slow, but use
// NVIDIA complex class => slow, but use
// define infinity etc. 
// microsoft gsl
// types.h use on device 

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

void fileWrite() {
	std::ofstream cx_minf("cx_min.txt");
	std::ofstream cy_minf("cy_min.txt");
	std::ofstream cx_maxf("cx_max.txt");
	std::ofstream cy_maxf("cy_max.txt");
	if (cx_minf.is_open() && cy_minf.is_open() && cx_maxf.is_open() && cy_maxf.is_open())
	{
		cy_maxf << "{";
		cx_maxf << "{";
		cy_minf << "{";
		cx_minf << "{";
		bool init{ true };

		for each (auto const factor in ZOOM_FACTORS)
		{
			cx_minf << (!init ? ", " : "") << (point_x + cx_min_factor * factor);
			cy_minf << (!init ? ", " : "") << (point_y + cy_min_factor * factor);
			cx_maxf << (!init ? ", " : "") << (point_x + cx_max_factor * factor);
			cy_maxf << (!init ? ", " : "") << (point_y + cy_max_factor * factor);

			init = false;
		}
		cy_maxf << "};";
		cx_maxf << "};";
		cy_minf << "};";
		cx_minf << "};";


		cy_maxf.close();
		cx_maxf.close();
		cy_minf.close();
		cx_minf.close();
	}
}

//(cx_min + x / (WIDTH_FACTOR) * (cx_max - cx_min)),
//(cy_min + y / (HEIGHT_FACTOR) * (cy_max - cy_min))

void factorWrite() {
	std::ofstream cx_f("cx_f.txt");
	std::ofstream cy_f("cy_f.txt");

	if (cx_f.is_open() && cy_f.is_open())
	{
		cx_f << "{";
		cy_f << "{";

		bool init{ true };

		for (int i{ 0 }; i < (sizeof(CX_MAX) / sizeof(float)); ++i)
		{
			cx_f << (!init ? ", " : "") << ((CX_MAX[i] - CX_MIN[i]) / WIDTH_FACTOR);
			cy_f << (!init ? ", " : "") << ((CY_MAX[i] - CY_MIN[i]) / HEIGHT_FACTOR);

			init = false;
		}
		cx_f << "};";
		cy_f << "};";

		cx_f.close();
		cy_f.close();
	}
}



int main() {

	if (!foundDevice()) {
		std::cout << "!!!!!!!!!! FAILED, no device found !!!!!!!!!!" << std::endl;
	}

	auto const images{ 200 };
	auto const cores{ 8 };
	auto const WORK_PER_CORE{ images / cores };
	auto const chunk_size_bitmap{ 50 }; // 70 sometimes out of memory

	// best is 1/4 row
	auto const BEST_INNER_SIZE{ WIDTH / 4 };

	init();

	//factorWrite();

	std::vector<float> times;
	std::vector<std::string> labels;

	times.emplace_back(measureTime(global_sequential_local_prallel, images, BEST_INNER_SIZE));
	labels.emplace_back("GS_LP_1.4");

	times.emplace_back(measureTime(global_parallel_local_parallel, images, BEST_INNER_SIZE, WORK_PER_CORE));
	labels.emplace_back("GP_LP_1.4");

	times.emplace_back(measureTime(global_parallel_local_parallel, images, WIDTH / 2, WORK_PER_CORE));
	labels.emplace_back("GP_LP_W2");

	times.emplace_back(measureTime(global_parallel_local_parallel, images, WIDTH, WORK_PER_CORE));
	labels.emplace_back("GP_LP_W");

	times.emplace_back(measureTime(global_parallel_local_parallel, images, BEST_INNER_SIZE, images / 2));
	labels.emplace_back("GP_LP_half");

	times.emplace_back(measureTime(global_parallel_local_sequential, images, WORK_PER_CORE));
	labels.emplace_back("GP_LS");	
	
	times.emplace_back(measureTime(parallel_gpu_byte_all_opt, images));
	labels.emplace_back("GP_LS");

	/*times.emplace_back(measureTime(global_sequential_local_sequential, images));
	labels.emplace_back("GS_LS");*/

	printResult(images, times, labels);

	return 0;
}