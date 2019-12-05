#include <iostream>
#include <vector>
#include "mandel.h"
#include "pfc_bitmap_3.h"
#include "pfc_threading.h"
#include "kernel.cuh"

// graphics card config
auto constexpr THREADS{ 1024 };
auto constexpr BLOCKS{ 36864 };
auto constexpr USE_TASKS{ true };
auto constexpr USE_THREADS{ false };

pfc::bitmap bitmaps[CPU_PARALLEL_SIZE];

void initCPU() {
	std::cout << "Allocating Bitmaps" << std::endl;
	static constexpr auto max_images{ CPU_PARALLEL_SIZE };
	pfc::parallel_range(USE_TASKS, 10, max_images, [](int const o, int const begin, int const end) {
		for (auto i{ begin }; i < end; ++i) {
			bitmaps[i] = pfc::bitmap{ WIDTH, HEIGHT };
			auto data{ bitmaps[i].pixel_span().data() };
			for (auto i{ 0 }; i < (PIXEL_PER_IMAGE); ++i) {
				data[i] = { 0, 0, 0 };
			}
		}
		});
	std::cout << "Bitmaps Allocated" << std::endl;
}

// CPU functions
pfc::byte_t color(float const cr, float const ci) {
	auto iterations{ START_ITERATION };
	auto zr{ 0.0f },
		zi{ 0.0f },
		zr_sq{ 0.0f },
		zi_sq{ 0.0f },
		temp_i{ 0.0f };

	do {
		zr_sq = zr * zr;
		zi_sq = zi * zi;
		temp_i = zr * zi;
		zi = temp_i + temp_i + ci;
		zr = zr_sq - zi_sq + cr;

	} while (zr_sq + zi_sq < BORDER && --iterations);

	return COLORS[iterations];
}

void global_sequential_local_sequential(int const images) {
	for (auto o{ 0 }; o < images; ++o)
	{
		auto const data{ bitmaps[o].pixel_span().data() };
		for (auto i{ 0 }; i < PIXEL_PER_IMAGE; ++i) {
			data[i] = { 0, 0, color((i % WIDTH) * X_FACTORS[o] + CX_MIN[o], (i / WIDTH) * Y_FACTORS[o] + CY_MIN[o]) };
		}
	}
}

void global_parallel_local_sequential(int const images, bool const useTasks) {

	pfc::parallel_range(useTasks, CPU_PARALLEL_SIZE, images, [images](int const idx, int const begin, int const end) {

		auto const data{ bitmaps[idx].pixel_span().data() };

		for (auto o{idx}; o < images; o += CPU_PARALLEL_SIZE) // images per thread
		{
			for (auto i{ 0 }; i < PIXEL_PER_IMAGE; ++i) { 
				data[i] = { 0, 0, color((i % WIDTH) * X_FACTORS[o] + CX_MIN[o], (i / WIDTH) * Y_FACTORS[o] + CY_MIN[o]) };
			}
		}
	});
}

void global_parallel_local_parallel(int const images, int const no_of_tasks, bool const useTasks) {

	pfc::parallel_range(useTasks, CPU_PARALLEL_SIZE, images, [no_of_tasks, useTasks, images](int const thread_idx, int const begin, int const end) {
		auto data{ bitmaps[thread_idx].pixel_span().data() };

		for (auto o{ thread_idx }; o < images; o += CPU_PARALLEL_SIZE) { // for image in thread

			pfc::parallel_range(useTasks, no_of_tasks, PIXEL_PER_IMAGE, [data, o](int innerIdx, int const begin, int const end) {
				for (auto i{ begin }; i < end; ++i) {
					data[i].red = color((i % WIDTH) * X_FACTORS[o] + CX_MIN[o], (i / WIDTH) * Y_FACTORS[o] + CY_MIN[o]);
				}
			});
		}
	});
}

void global_sequential_local_prallel(int const images, int const no_of_tasks, bool const useTasks) {

	auto image_idx{ 0 };
	for (auto o{ 0 }; o < images; ++o)
	{
		image_idx %= CPU_PARALLEL_SIZE;
		auto const data{ bitmaps[image_idx++].pixel_span().data() };

		pfc::parallel_range(useTasks, no_of_tasks, PIXEL_PER_IMAGE, [data, o](int innerIdx, int const begin, int const end) {
			for (auto i{ begin }; i < end; ++i) {
				data[i] = { 0, 0, color((i % WIDTH) * X_FACTORS[o] + CX_MIN[o], (i / WIDTH) * Y_FACTORS[o] + CY_MIN[o]) };
			}
			});
	}
}

void global_parallel_local_sequential_task(int const images) {
	global_parallel_local_sequential(images, USE_TASKS);
}

void global_parallel_local_sequential_thread(int const images) {
	global_parallel_local_sequential(images, USE_THREADS);
}

void global_parallel_local_parallel_task(int const images, int const no_of_tasks) {
	global_parallel_local_parallel(images, no_of_tasks, USE_TASKS);
}

void global_parallel_local_parallel_thread(int const images, int const no_of_threads) {
	global_parallel_local_parallel(images, no_of_threads, USE_THREADS);
}

void global_sequential_local_prallel_task(int const images, int const no_of_tasks) {
	global_sequential_local_prallel(images, no_of_tasks, USE_TASKS);
}

void global_sequential_local_prallel_thread(int const images, int const no_of_threads) {
	global_sequential_local_prallel(images, no_of_threads, USE_THREADS);
}

void storeLastImage(int const calculated_images, std::string const& prefix) {
	bitmaps[CPU_PARALLEL_SIZE - 1].to_file("../img/" + prefix + "_" + std::to_string(calculated_images) + ".bmp");
	std::cout << "stored last image for " << prefix << std::endl;
}

void storeImages(int const calculated_images, std::string const& prefix) {
	for (auto i{ 0 }; i < CPU_PARALLEL_SIZE; ++i) {
		bitmaps[i].to_file("../img/" + prefix + "_" + std::to_string(calculated_images - CPU_PARALLEL_SIZE + i + 1) + ".bmp");
	}
	std::cout << "stored images for " << prefix << std::endl;
}

// GPU
auto constexpr DEVICE_SIZE{ 20 };
auto constexpr MEMORY_SIZE{ PIXEL_PER_IMAGE * sizeof(pfc::BGR_4_t) };
std::vector<pfc::BGR_4_t*> bgrDevicePointers;
cudaStream_t streams[CPU_PARALLEL_SIZE];

void sequential_gpu(int const images) {
	
	for (auto o{ 0 }; o < images; ++o)
	{
		call_mandel_kernel(BLOCKS, THREADS, bgrDevicePointers[0], PIXEL_PER_IMAGE, o);

		gpuErrchk(cudaPeekAtLastError());

		gpuErrchk(cudaMemcpy(bitmaps[CPU_PARALLEL_SIZE - 1].pixel_span().data(), bgrDevicePointers[0], MEMORY_SIZE, cudaMemcpyDeviceToHost));
	}
}

void freeGPU() {
	gpuErrchk(cudaDeviceReset());
	std::cout << "Freed GPU memory" << std::endl;
}


void initGPU() {
	std::cout << "Alloc GPU memory" << std::endl;

	for (auto i{ 0 }; i < DEVICE_SIZE; ++i) {
		pfc::BGR_4_t* dp_destination_bgr{ nullptr };
		gpuErrchk(cudaMalloc(&dp_destination_bgr, MEMORY_SIZE));
		bgrDevicePointers.emplace_back(dp_destination_bgr);
		gpuErrchk(cudaStreamCreate(&streams[i]));
	}

	std::cout << "Allocated GPU memory" << std::endl;
}

auto constexpr no_of_images_per_stream{ 200 / CPU_PARALLEL_SIZE };
auto constexpr no_of_streams{ CPU_PARALLEL_SIZE };

void parallel_streamed_GPU_prallel_range(int const images) {

	pfc::parallel_range(USE_TASKS, DEVICE_SIZE, DEVICE_SIZE, [](int t, int begin, int end) {
		call_mandel_kernel(BLOCKS, THREADS, bgrDevicePointers[t], PIXEL_PER_IMAGE, t, streams[t]);
		gpuErrchk(cudaPeekAtLastError());

		gpuErrchk(cudaMemcpyAsync(bitmaps[t].pixel_span().data(), bgrDevicePointers[t], MEMORY_SIZE, cudaMemcpyDeviceToHost, streams[t]));

		gpuErrchk(cudaStreamSynchronize(streams[t]));
		});
}

void parallel_streamed_GPU_for_loop(int const images) {
	

	for (int t = 0; t < 20; ++t) {
		call_mandel_kernel(BLOCKS, THREADS, bgrDevicePointers[t], PIXEL_PER_IMAGE, t, streams[t]);
		gpuErrchk(cudaPeekAtLastError());

		gpuErrchk(cudaMemcpyAsync(bitmaps[t].pixel_span().data(), bgrDevicePointers[t], MEMORY_SIZE, cudaMemcpyDeviceToHost, streams[t]));

		gpuErrchk(cudaStreamSynchronize(streams[t]));
	};
}

void parallel_GPU_stream0(int const images) {
	pfc::parallel_range(USE_TASKS, DEVICE_SIZE, DEVICE_SIZE, [](int t, int begin, int end) {
		call_mandel_kernel(BLOCKS, THREADS, bgrDevicePointers[t], PIXEL_PER_IMAGE, t);
		gpuErrchk(cudaPeekAtLastError());

		gpuErrchk(cudaMemcpy(bitmaps[t].pixel_span().data(), bgrDevicePointers[t], MEMORY_SIZE, cudaMemcpyDeviceToHost));

		gpuErrchk(cudaStreamSynchronize(streams[t]));
	});
}