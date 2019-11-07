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
#include <iostream>
// TODO:
// CPU threads vs task with param settings when is what better
// try to be creative with paralelism on CPU, e.g. 4 parallel images which are calculated parallel
// c++ complex class => slow, but use
// NVIDIA complex class => slow, but use
// define infinity etc. 
// microsoft gsl
// types.h use on device 

bool foundDevice() {
	auto count{ 0 };
	gpuErrchk(cudaGetDeviceCount(&count));
	if (count < 1) return false;

	auto device_no{ 0 };
	gpuErrchk(cudaSetDevice(device_no));

	return true;
}

int calcAsyncFunction(float const& point_x, float const& point_y, int const thread_number, int const iterations, std::string const& prefix) {
	auto zoom_factor{ 1.0f };
	// calculate zoom_factor
	for (auto i{ 1 }; i < thread_number; ++i) {
		zoom_factor *= ZOOM_FACTOR;
	}
	// set field of view
	auto const cx_min{ point_x + cx_min_factor * zoom_factor };
	auto const cx_max{ point_x + cx_max_factor * zoom_factor };
	auto const cy_min{ point_y + cy_min_factor * zoom_factor };
	auto const cy_max{ point_y + cy_max_factor * zoom_factor };

	// create the bitmap
	pfc::bitmap const bmp{ WIDTH, HEIGHT };
	auto s{ bmp.pixel_span().begin() };

	// foreach point
	for (auto i{ 0 }; i < HEIGHT; ++i) {
		for (auto j{ 0 }; j < WIDTH; ++j) {
			//calculate the color value
			*s = { 0, 0, valueHost(j, i, cx_min, cy_min, cx_max, cy_max, iterations) };
			++s;
		}
	}
	// store image
	bmp.to_file("./" + prefix + "_iter_" + std::to_string(thread_number) + ".bmp");

	return 0;
}

int calcAsyncFunction2(std::unique_ptr<pfc::byte_t[]>& dest, int size, int const& pixel_per_thread,
	float const& cx_min, float const& cx_max, float const& cy_min, float const& cy_max, int const start, int const end) {

	auto x{ start % WIDTH };
	auto y{ start / WIDTH };

	// copy at thread position if the position is valid
	if (start < size) {
		for (auto i{ start }; i < end; ++i) {
			dest[i] = valueDevice(x++, y, cx_min, cy_min, cx_max, cy_max, ITERATIONS);
		}
	}
	return 0;
}

int calcAsyncFunction3(pfc::bitmap& dest, int size, int const& pixel_per_thread,
	float const& cx_min, float const& cx_max, float const& cy_min, float const& cy_max, int const start, int const end) {

	auto x{ start % WIDTH };
	auto y{ start / WIDTH };

	auto s{ dest.pixel_span().begin() };
	std::advance(s, start);
	// copy at thread position if the position is valid
	if (start < size) {
		for (auto i{ start }; i < end; ++i) {
			*s = { 0, 0, valueDevice(x++, y, cx_min, cy_min, cx_max, cy_max, ITERATIONS) };
			++s;
		}
	}
	return 0;
}

int calcAsyncFunctionLoopUnroll(float const& point_x,
	float const& point_y,
	int const thread_number,
	int const iterations,
	std::string const& prefix,
	int const innerTasksPixels)
{
	auto zoom_factor{ 1.0f };
	// calculate zoom_factor
	for (auto i{ 1 }; i < thread_number; ++i) {
		zoom_factor *= ZOOM_FACTOR;
	}
	// set field of view
	auto const cx_min{ point_x + cx_min_factor * zoom_factor };
	auto const cx_max{ point_x + cx_max_factor * zoom_factor };
	auto const cy_min{ point_y + cy_min_factor * zoom_factor };
	auto const cy_max{ point_y + cy_max_factor * zoom_factor };

	std::vector<std::future<int>> tasks;
	auto dest{ std::make_unique<pfc::byte_t[]>(PIXEL_PER_IMAGE * sizeof(pfc::byte_t)) };

	for (auto i{ 0 }; i < PIXEL_PER_IMAGE; i += innerTasksPixels) {
		tasks.emplace_back(std::async(
			std::launch::async,
			&calcAsyncFunction2, std::ref(dest), PIXEL_PER_IMAGE, innerTasksPixels,
			cx_min, cx_max, cy_min, cy_max, i, (i + innerTasksPixels)));
	}

	// wait for all points
	for (auto i{ 0 }; i < tasks.size(); ++i) {
		tasks[i].wait();
	}

	pfc::bitmap const bmp{ WIDTH, HEIGHT };
	auto s{ bmp.pixel_span().begin() };
	for (auto i{ 0 }; i < PIXEL_PER_IMAGE; ++i) {
		*s = { 0, 0, dest[i] };
		++s;
	}

	// store image
	bmp.to_file("./" + prefix + "_iter_" + std::to_string(thread_number) + ".bmp");

	return 0;
}

int calcAsyncFunctionLoopUnroll2(float const& point_x, float const& point_y, int const thread_number, int const iterations, std::string const& prefix, int const innerTasksPixels) {
	auto zoom_factor{ 1.0f };
	// calculate zoom_factor
	for (auto i{ 1 }; i < thread_number; ++i) {
		zoom_factor *= ZOOM_FACTOR;
	}
	// set field of view
	auto const cx_min{ point_x + cx_min_factor * zoom_factor };
	auto const cx_max{ point_x + cx_max_factor * zoom_factor };
	auto const cy_min{ point_y + cy_min_factor * zoom_factor };
	auto const cy_max{ point_y + cy_max_factor * zoom_factor };

	std::vector<std::future<int>> tasks;
	pfc::bitmap bmp{ WIDTH, HEIGHT };

	for (auto i{ 0 }; i < PIXEL_PER_IMAGE; i += innerTasksPixels) {
		tasks.emplace_back(std::async(
			std::launch::async,
			&calcAsyncFunction3, std::ref(bmp), PIXEL_PER_IMAGE, innerTasksPixels,
			cx_min, cx_max, cy_min, cy_max, i, (i + innerTasksPixels)));
	}

	// wait for all points
	for (auto i{ 0 }; i < tasks.size(); ++i) {
		tasks[i].wait();
	}

	// store image
	bmp.to_file("./" + prefix + "_iter_" + std::to_string(thread_number) + ".bmp");

	return 0;
}

// code to be executed on host
void runOnHostOneThreadPerImage(int const images) {
	std::vector<std::thread> threads;
	std::string const prefix{ "th" };

	// one thread foreach image
	for (auto i{ 0 }; i < images; ++i) {
		threads.emplace_back(std::thread(&calcAsyncFunction, point_x, point_y, i + 1, ITERATIONS, std::ref(prefix)));
	}

	// wait for all points
	for (int i = 0; i < images; ++i) {
		threads[i].join();
	}
}

// code to be executed on host
void runOnHostOneThreadPerImagePfc(int const images) {
	std::string const prefix{ "th" };

	pfc::parallel_range(true, images, images, [prefix](int idx, int begin, int end) {
		calcAsyncFunction(point_x, point_y, idx, ITERATIONS, prefix);
		});
}

// code to be executed on host
void runOnHostOneTaskPerImage(int const images) {
	std::vector<std::future<int>> tasks;
	std::string const prefix{ "ta" };

	// one thread foreach image
	for (auto i{ 0 }; i < images; ++i) {
		tasks.emplace_back(std::async(std::launch::async, &calcAsyncFunction, point_x, point_y, i + 1, ITERATIONS, std::ref(prefix)));
	}

	// wait for all points
	for (int i = 0; i < images; ++i) {
		tasks[i].wait();
	}
}

// code to be executed on host
void runOnHostOneTaskPerImageLoopUnroll(int const images, int const innerTasksPixels) {
	std::vector<std::future<int>> tasks;
	std::string const prefix{ "ur" };

	// one thread foreach image
	for (auto i{ 0 }; i < images; ++i) {
		tasks.emplace_back(std::async(std::launch::async, &calcAsyncFunctionLoopUnroll, point_x, point_y, i + 1, ITERATIONS, std::ref(prefix), innerTasksPixels));
	}

	// wait for all points
	for (int i = 0; i < images; ++i) {
		tasks[i].wait();
	}
}

// code to be executed on host
void runOnHostOneTaskPerImageLoopUnroll2(int const images, int const innerTasksPixels) {
	std::vector<std::future<int>> tasks;
	std::string const prefix{ "u2" };

	// one thread foreach image
	for (auto i{ 0 }; i < images; ++i) {
		tasks.emplace_back(std::async(std::launch::async, &calcAsyncFunctionLoopUnroll2, point_x, point_y, i + 1, ITERATIONS, std::ref(prefix), innerTasksPixels));
	}

	// wait for all points
	for (int i = 0; i < images; ++i) {
		tasks[i].wait();
	}
}


int calculateOnDevice(float const& point_x, float const& point_y, int const current_iteration, int const iterations, float const& pixel_per_thread) {
	auto zoom_factor{ 1.0f };
	// calculate zoom_factor
	for (auto i{ 1 }; i < current_iteration; ++i) {
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
	auto s{ bmp.pixel_span().begin() };
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
		*s = { 0, 0, hp_destination[i] };
		++s;
	}
	// store test image
	//bmp.to_file("./gpu_iter_" + std::to_string(current_iteration) + ".bmp");

	return 0;
}

int calculateOnDeviceWholeBitmap(float const& point_x, float const& point_y, int const current_iteration, int const iterations, float const& pixel_per_thread) {
	auto zoom_factor{ 1.0f };
	// calculate zoom_factor
	for (auto i{ 1 }; i < current_iteration; ++i) {
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
	//bmp.to_file("./gpu_whbi_" + std::to_string(current_iteration) + ".bmp");

	return 0;
}

void runOnDevice(int const images, float const& pixel_per_thread) {
	std::vector<std::future<int>> tasks;
	std::string const prefix{ "ur" };

	// one task for each image
	for (auto i{ 0 }; i < images; ++i) {
		tasks.emplace_back(std::async(std::launch::async, &calculateOnDevice, point_x, point_y, i + 1, ITERATIONS, pixel_per_thread));
	}

	// wait for completion
	for (int i = 0; i < images; ++i) {
		tasks[i].wait();
	}
}

void runOnDeviceWholeBitmap(int const images, float const& pixel_per_thread) {
	std::vector<std::future<int>> tasks;
	std::string const prefix{ "ur" };

	// one task for each image
	for (auto i{ 0 }; i < images; ++i) {
		tasks.emplace_back(std::async(std::launch::async, &calculateOnDeviceWholeBitmap, point_x, point_y, i + 1, ITERATIONS, pixel_per_thread));
	}

	// wait for completion
	for (int i = 0; i < images; ++i) {
		tasks[i].wait();
	}
}

void global_parallel_local_parallel(int const images, int const inner_size) {
	// one thread per image
	pfc::parallel_range(true, images, images, [inner_size](int outerIdx, int begin, int end) {
		pfc::bitmap const bmp{ WIDTH, HEIGHT };

		// calculate zoom_factor
		auto zoom_factor{ 1.0f };
		for (auto i{ 1 }; i < outerIdx; ++i) {
			zoom_factor *= ZOOM_FACTOR;
		}

		// set field of view / Zoom in
		auto const cx_min{ point_x + cx_min_factor * zoom_factor };
		auto const cx_max{ point_x + cx_max_factor * zoom_factor };
		auto const cy_min{ point_y + cy_min_factor * zoom_factor };
		auto const cy_max{ point_y + cy_max_factor * zoom_factor };

		// one task calcualtes inner_size pixels
		pfc::parallel_range(true, inner_size, PIXEL_PER_IMAGE, [bmp, cx_min, cx_max, cy_min, cy_max](int innerIdx, int begin, int end) {
			// get the current line in the image
			auto y{ begin / WIDTH };
			// advance to the current starting pixel
			auto s{ bmp.pixel_span().begin() };
			std::advance(s, begin);
			// copy at thread position if the position is valid
			if (begin < end) {
				for (auto i{ begin }; i < end; ++i) {
					*s = { 0, 0, valueDevice(i, y, cx_min, cy_min, cx_max, cy_max, ITERATIONS) };
					++s;
				}
			}
			});

		bmp.to_file("./cpu_gp_lp" + std::to_string(outerIdx) + ".bmp");
		});
}

void global_parallel_local_sequential(int const images) {
	// one thread per image
	pfc::parallel_range(true, images, images, [](int outerIdx, int begin, int end) {
		pfc::bitmap const bmp{ WIDTH, HEIGHT };

		// calculate zoom_factor
		auto zoom_factor{ 1.0f };
		for (auto i{ 1 }; i < outerIdx; ++i) {
			zoom_factor *= ZOOM_FACTOR;
		}

		// set field of view / Zoom in
		auto const cx_min{ point_x + cx_min_factor * zoom_factor };
		auto const cx_max{ point_x + cx_max_factor * zoom_factor };
		auto const cy_min{ point_y + cy_min_factor * zoom_factor };
		auto const cy_max{ point_y + cy_max_factor * zoom_factor };

		// advance to the current starting pixel
		auto s{ bmp.pixel_span().begin() };
	
		// foreach pixel in image
		for (auto i{ 0 }; i < PIXEL_PER_IMAGE; ++i) {
			auto x{ i % WIDTH };
			auto y{ i / WIDTH };

			*s = { 0, 0, valueDevice(x, y, cx_min, cy_min, cx_max, cy_max, ITERATIONS) };
			++s;
		}
		
		bmp.to_file("./cpu_gp_ls" + std::to_string(outerIdx) + ".bmp");
	});
}

int main() {

	if (!foundDevice()) {
		std::cout << "!!!!!!!!!! FAILED, no device found !!!!!!!!!!" << std::endl;
	}

	std::vector<float> zoom_factor;
	auto factor{ ZOOM_FACTOR };

	for (auto i{ 0 }; i < 200; ++i) {
		zoom_factor.push_back(factor);
		factor *= ZOOM_FACTOR;
	}


	auto images{ 1 };

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
	//	measureTime(global_parallel_local_parallel, images, WIDTH), "", "CPU GP_LP", images);	
	
	
	//printResult(200,
	//	measureTime(global_parallel_local_sequential, images), "", "CPU GP_LS", images);



	//measureTime(runOnHostOneThreadPerImagePfc, 1);

	//measureTime(runOnDeviceWholeBitmap, images, 0);

	//pfc::parallel_range_task(2, 2, [](int idx, int begin, int end) {
	//	std::cout << idx << ", " << begin << ", " << end << " ";
	//	});



	return 0;
}