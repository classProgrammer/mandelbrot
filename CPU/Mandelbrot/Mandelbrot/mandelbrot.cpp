#include <cstdlib>
#include <complex>
#include <fstream> // for files manipulation
#include <complex> // for complex numbers
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

// define to store the bmp files
#define STOREIMAGES

// precalculated indices for value mapping
std::vector<int> X_VAL;
std::vector<int> Y_VAL;

pfc::byte_t valueHost_opt(int const inner_idx, int const outer_index) {
	// calculate the constant
	pfc::complex<float> c(
		(CX_MIN[outer_index] + X_VAL[inner_idx] / (WIDTH_FACTOR) * (CX_MAX[outer_index] - CX_MIN[outer_index])),
		(CY_MIN[outer_index] + Y_VAL[inner_idx] / (HEIGHT_FACTOR) * (CY_MAX[outer_index] - CY_MIN[outer_index]))
	);
	// initialize z
	pfc::complex<float> z(0.0f, 0.0f);
	auto iterations{ 0 };
	// calculate z
	while (z.norm() < 4 && iterations++ < ITERATIONS) {
		z = z.square() + c;
	}
	// set color gradient
	return iterations < ITERATIONS ? COLORS[iterations] : 0;
}
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

// CPU functions
void global_sequential_local_sequential(int const images) {

	for (auto o{ 0 }; o < images; ++o)
	{
		pfc::bitmap const bmp{ WIDTH, HEIGHT };
		auto data{ bmp.pixel_span().data() };

		// foreach pixel in image
		for (auto i{ 0 }; i < PIXEL_PER_IMAGE; ++i) {
			data[i] = { 0, 0, valueHost(X_VAL[i], Y_VAL[i], CX_MIN[o], CY_MIN[o], CX_MAX[o], CY_MAX[o], ITERATIONS) };
		}
#ifdef STOREIMAGES
		bmp.to_file("../img/cpu_gs_ls" + std::to_string(o + 1) + ".bmp");
#endif // STOREIMAGES
	}
}

void global_parallel_local_sequential(int const images, int const outer_size) {
	// one thread per image
	pfc::parallel_range(true, outer_size, images, [](int outerIdx, int begin, int end) {
		pfc::bitmap const bmp{ WIDTH, HEIGHT };
		auto data{ bmp.pixel_span().data() };

		// foreach pixel in image
		for (auto i{ 0 }; i < PIXEL_PER_IMAGE; ++i) {
			data[i] = { 0, 0, valueHost(X_VAL[i], Y_VAL[i], CX_MIN[outerIdx], CY_MIN[outerIdx], CX_MAX[outerIdx], CY_MAX[outerIdx], ITERATIONS) };
		}
#ifdef STOREIMAGES
		bmp.to_file("../img/cpu_gp_ls" + std::to_string(outerIdx + 1) + ".bmp");
#endif // STOREIMAGES
		});
}

void global_parallel_local_parallel(int const images, int const inner_size, int const outer_size) {
	// one thread per image
	pfc::parallel_range(true, outer_size, images, [inner_size](int thread_idx, int begin, int end) {
		// o = outer
		// i = inner
		for (auto o{ begin }; o < end; ++o) {
			pfc::bitmap const bmp{ WIDTH, HEIGHT };
			auto data{ bmp.pixel_span().data() };

			// foreach pixel in image
			pfc::parallel_range(true, inner_size, PIXEL_PER_IMAGE, [data, o](int innerIdx, int begin, int end) {
				for (auto i{ begin }; i < end; ++i) {
					data[i] = { 0, 0, valueHost(X_VAL[i], Y_VAL[i], CX_MIN[o], CY_MIN[o], CX_MAX[o], CY_MAX[o], ITERATIONS) };
				}
				});
#ifdef STOREIMAGES
			bmp.to_file("../img/cpu_gp_lp" + std::to_string(o + 1) + ".bmp");
#endif // STOREIMAGES
		}
	});
}

void global_parallel_local_parallel_v2(int const images, int const inner_size, int const outer_size) {
	// one thread per image
	pfc::parallel_range(true, outer_size, images, [inner_size](int thread_idx, int begin, int end) {
		// o = outer
		// i = inner
		for (auto o{ begin }; o < end; ++o) {
			pfc::bitmap const bmp{ WIDTH, HEIGHT };
			auto data{ bmp.pixel_span().data() };

			// foreach pixel in image
			pfc::parallel_range(true, inner_size, PIXEL_PER_IMAGE, [data, o](int innerIdx, int begin, int end) {
				for (auto i{ begin }; i < end; ++i) {
					data[i] = { 0, 0, valueHost_opt(i, o) };
				}
				});
#ifdef STOREIMAGES
			bmp.to_file("../img/cpu_gp_lp_v2" + std::to_string(o + 1) + ".bmp");
#endif // STOREIMAGES
		}
		});
}

void global_sequential_local_prallel(int const images, int const inner_size) {
	// one thread per image
	for (auto o{ 0 }; o < images; ++o)
	{
		pfc::bitmap const bmp{ WIDTH, HEIGHT };
		auto data{ bmp.pixel_span().data() };

		// foreach pixel in image
		pfc::parallel_range(true, inner_size, PIXEL_PER_IMAGE, [data, o](int innerIdx, int begin, int end) {
			for (auto i{ begin }; i < end; ++i) {
				data[i] = { 0, 0, valueHost(X_VAL[i], Y_VAL[i], CX_MIN[o], CY_MIN[o], CX_MAX[o], CY_MAX[o], ITERATIONS) };
			}
		});
#ifdef STOREIMAGES
		bmp.to_file("../img/cpu_gs_lp" + std::to_string(o + 1) + ".bmp");
#endif // STOREIMAGES
	}
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

void precalculateIndices() {
	for (auto i{ 0 }; i < PIXEL_PER_IMAGE; ++i) {
		auto x{ i % WIDTH };
		auto y{ i / WIDTH };

		X_VAL.emplace_back(x);
		Y_VAL.emplace_back(y);
	}
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

int main() {

	if (!foundDevice()) {
		std::cout << "!!!!!!!!!! FAILED, no device found !!!!!!!!!!" << std::endl;
	}

	auto const images{ 200 };
	auto const cores{ 8 };
	auto const work_per_core{ images / cores };
	auto const chunk_size_bitmap{ 50 }; // 70 sometimes out of memory

	precalculateIndices();

	//printResult(measureTime(runOnHostOneTaskPerImageLoopUnroll2, images, 1024),
	//	measureTime(runOnHostOneTaskPerImageLoopUnroll2, images, WIDTH), "CPU LUR 1024", "CPU LUR WIDTH", images);

	//printResult(measureTime(runOnHostOneTaskPerImageLoopUnroll2, images, 2048),
	//	measureTime(runOnHostOneTaskPerImageLoopUnroll2, images, 4096), "CPU LUR 2048", "CPU LUR 4096", images);


	//printResult(measureTime(runOnHostOneTaskPerImageLoopUnroll2, images, WIDTH),
	//	measureTime(runOnHostOneTaskPerImage, images), "CPU LUR_WIDTH", "CPU 8 Threads", images);
	//measureTime(runOnDevice, images, 0);
	//printResult(200,
	//	measureTime(runOnDevice, images, 0), "", "GPU Whole Bitmap", images);


	//printResult(200,
	//	measureTime(global_parallel_local_parallel, images, WIDTH, images), "", "CPU GP_LP", images);	


	//printResult(200,
	//	measureTime(global_parallel_local_sequential, images), "", "CPU GP_LS", images);

	//printResult(200,
	//	measureTime(global_sequential_local_prallel, images, WIDTH), "", "CPU GS_LP", images);

	//printResult(measureTime(global_parallel_local_sequential, images),
	//	measureTime(global_parallel_local_parallel, images, WIDTH, images), "CPU GP_LS", "CPU GP_LP", images);	
	
	//printResult(measureTime(global_parallel_local_parallel, images, WIDTH / 4, images),
	//	measureTime(global_parallel_local_parallel, images, WIDTH / 2, work_per_core), "CPU imgs", "CPU 8 core", images);		
	
	//printResult(measureTime(global_parallel_local_parallel_v2, images, WIDTH / 2, work_per_core),
	//	measureTime(global_parallel_local_parallel, images, WIDTH / 2, work_per_core), "CPU 8 opt", "CPU 8 no opt", images);	
	
	//printResult(measureTime(parallel_gpu_byte_all, images),
	//	measureTime(parallel_gpu_bitmap_chunked, images, chunk_size_bitmap), "GPU Byte all", "GPU Bitmap chunked", images);	
	
	
	printResult(100,
		measureTime(parallel_gpu_byte_all_opt, images), "GPU Byte no opt", "GPU Byte opt", images);

	return 0;
}