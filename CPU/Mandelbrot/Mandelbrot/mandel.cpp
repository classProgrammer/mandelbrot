#include <iostream>
#include <vector>
#include "mandel.h"
#include "pfc_bitmap_3.h"
#include "pfc_threading.h"
#include "kernel.cuh"

// graphics card config
auto constexpr THREADS{ 1024 };
auto constexpr BLOCKS{ 36864 };

pfc::bitmap bitmaps[CPU_PARALLEL_SIZE];

void initCPU() {
	std::cout << "Allocating Bitmaps" << std::endl;
	static constexpr auto max_images{ CPU_PARALLEL_SIZE };
	pfc::parallel_range(true, 10, max_images, [](int const o, int const begin, int const end) {
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

void global_parallel_local_sequential_task(int const images) {
	
	pfc::parallel_range(true, CPU_PARALLEL_SIZE, images, [](int const o, int const begin, int const end) {

		auto const data{ bitmaps[o].pixel_span().data() };
		
		for (auto i{ 0 }; i < PIXEL_PER_IMAGE; ++i) {
			data[i] = { 0, 0, color((i % WIDTH) * X_FACTORS[o] + CX_MIN[o], (i / WIDTH) * Y_FACTORS[o] + CY_MIN[o]) };
		}
	});
}

void global_parallel_local_parallel_task(int const images, int const no_of_tasks) {
	
	pfc::parallel_range(true, CPU_PARALLEL_SIZE, images, [no_of_tasks](int const o, int const begin, int const end) {
		pfc::parallel_range(true, no_of_tasks, PIXEL_PER_IMAGE, [data{ bitmaps[o].pixel_span().data() }, o](int innerIdx, int const begin, int const end) {
			for (auto i{ begin }; i < end; ++i) {
				data[i] = { 0, 0, color((i % WIDTH) * X_FACTORS[o] + CX_MIN[o], (i / WIDTH) * Y_FACTORS[o] + CY_MIN[o]) };
			}
		});
	});
}

void global_parallel_local_parallel_task2(int const images, int const no_of_tasks) {

	pfc::parallel_range(true, CPU_PARALLEL_SIZE, images, [no_of_tasks](int const thread_idx, int const begin, int const end) {

		for (auto o{ begin }; o < end; ++o) {
			pfc::parallel_range(true, no_of_tasks, PIXEL_PER_IMAGE, [data{ bitmaps[o].pixel_span().data() }, o](int innerIdx, int const begin, int const end) {
				for (auto i{ begin }; i < end; ++i) {
					data[i].red = color((i % WIDTH) * X_FACTORS[o] + CX_MIN[o], (i / WIDTH) * Y_FACTORS[o] + CY_MIN[o]);
				}
			});

		}
		});
}

//void global_parallel_local_parallel_task3(int const images, int const no_of_tasks) {
//	auto const tasks_per_img = no_of_tasks / images;
//	auto const one_unit = PIXEL_PER_IMAGE / tasks_per_img;
//
//	pfc::parallel_range_no_size(no_of_tasks, [tasks_per_img, one_unit](int const task_idx, int begin, int end ) {
//
//		auto const image_no{ task_idx / tasks_per_img };
//		begin = task_idx % tasks_per_img * one_unit;
//		end = begin + one_unit;
//
//		auto const data{ bitmaps[image_no].pixel_span().data() };
//
//		for (auto i{ begin }; i < end; ++i) {
//			data[i] = { 0, 0, color((i % WIDTH) * X_FACTORS[image_no] + CX_MIN[image_no], (i / WIDTH) * Y_FACTORS[image_no] + CY_MIN[image_no]) };
//		}
//	});
//}

void global_sequential_local_prallel_task(int const images, int const no_of_tasks) {
	for (auto o{ 0 }; o < images; ++o)
	{
		auto const data{ bitmaps[o].pixel_span().data() };
		
		pfc::parallel_range(true, no_of_tasks, PIXEL_PER_IMAGE, [data, o](int innerIdx, int const begin, int const end) {
			for (auto i{ begin }; i < end; ++i) {
				data[i] = { 0, 0, color((i % WIDTH) * X_FACTORS[o] + CX_MIN[o], (i / WIDTH) * Y_FACTORS[o] + CY_MIN[o]) };
			}
			});
	}
}

void global_parallel_local_sequential_thread(int const images) {
	
	pfc::parallel_range(false, images, images, [](int const o, int const begin, int const end) {

		
		auto const data{ bitmaps[o].pixel_span().data() };
		for (auto i{ 0 }; i < PIXEL_PER_IMAGE; ++i) {
			data[i] = { 0, 0, color((i % WIDTH) * X_FACTORS[o] + CX_MIN[o], (i / WIDTH) * Y_FACTORS[o] + CY_MIN[o]) };
		}

		});
}

void global_parallel_local_parallel_thread(int const images, int const no_of_threads) {
	pfc::parallel_range(false, images, images, [no_of_threads](int const o, int const begin, int const end) {
		auto const data{ bitmaps[o].pixel_span().data() };
		
		pfc::parallel_range(false, no_of_threads, PIXEL_PER_IMAGE, [data, o](int innerIdx, int const begin, int const end) {
			for (auto i{ begin }; i < end; ++i) {
				data[i] = { 0, 0, color((i % WIDTH) * X_FACTORS[o] + CX_MIN[o], (i / WIDTH) * Y_FACTORS[o] + CY_MIN[o]) };
			}
			});
		});
}

void global_sequential_local_prallel_thread(int const images, int const no_of_threads) {
	for (auto o{ 0 }; o < images; ++o)
	{
		auto const data{ bitmaps[o].pixel_span().data() };
		
		pfc::parallel_range(false, no_of_threads, PIXEL_PER_IMAGE, [data, o](int innerIdx, int const begin, int const end) {
			for (auto i{ begin }; i < end; ++i) {
				data[i] = { 0, 0, color((i % WIDTH) * X_FACTORS[o] + CX_MIN[o], (i / WIDTH) * Y_FACTORS[o] + CY_MIN[o]) };
			}
		});
	}
}

void storeLastImage(int const calculated_images, std::string const& prefix) {
	auto const i{CPU_PARALLEL_SIZE - 1 };
	bitmaps[i].to_file("../img/" + prefix + "_" + std::to_string(calculated_images) + ".bmp");
	std::cout << "stored last image for " << prefix << std::endl;
}

void storeImages(int const calculated_images, std::string const& prefix) {
	for (auto i{ 0 }; i < CPU_PARALLEL_SIZE; ++i) {
		bitmaps[i].to_file("../img/" + prefix + "_" + std::to_string(calculated_images - CPU_PARALLEL_SIZE + i + 1) + ".bmp");
	}

	std::cout << "stored images for " << prefix << std::endl;
}

// GPU
auto const deviceSize{ CPU_PARALLEL_SIZE };
auto const mappingSize{ 1 };

std::vector<pfc::BGR_4_t*> bgrDevicePointers;
std::vector<pfc::byte_t*> byteDevicePointers;

auto const memory_size_bgr{ PIXEL_PER_IMAGE * sizeof(pfc::BGR_4_t) };
auto const memory_size_byte{ PIXEL_PER_IMAGE * sizeof(pfc::byte_t) };

void calculateOnDeviceBitmap(int const iteration) {
	static auto const memory_size{ memory_size_bgr };

	call_mandel_kernel(BLOCKS, THREADS, bgrDevicePointers[0], PIXEL_PER_IMAGE, iteration);

	gpuErrchk(cudaPeekAtLastError());

	gpuErrchk(cudaMemcpy(bitmaps[CPU_PARALLEL_SIZE-1].pixel_span().data(), bgrDevicePointers[0], memory_size, cudaMemcpyDeviceToHost));
}

void calculateOnDeviceByte(int const iteration) {
	static auto const memory_size{ memory_size_byte };

	auto hp_destination{ std::make_unique<pfc::byte_t[]>(memory_size) };

	call_mandel_kernel(BLOCKS, THREADS, byteDevicePointers[iteration], PIXEL_PER_IMAGE, iteration);

	gpuErrchk(cudaPeekAtLastError());
		
	gpuErrchk(cudaMemcpy(hp_destination.get(), byteDevicePointers[iteration], memory_size, cudaMemcpyDeviceToHost));

	auto const data{ bitmaps[iteration].pixel_span().data() };
	// copy to bmp
	for (auto i{ 0 }; i < PIXEL_PER_IMAGE; ++i) {
		data[i] = { 0, 0, hp_destination[i] };
	}
}

void calculateOnDeviceBitmap_parallel(int const iteration) {
	static auto const memory_size{ memory_size_bgr };

	call_mandel_kernel(BLOCKS, THREADS, bgrDevicePointers[0], PIXEL_PER_IMAGE, iteration);

	gpuErrchk(cudaPeekAtLastError());

	gpuErrchk(cudaMemcpy(bitmaps[iteration].pixel_span().data(), bgrDevicePointers[iteration], memory_size, cudaMemcpyDeviceToHost));
}

void calculateOnDeviceByte_parallel(int const iteration) {
	static auto const memory_size{ memory_size_byte };

	auto hp_destination{ std::make_unique<pfc::byte_t[]>(memory_size) };

	call_mandel_kernel(BLOCKS, THREADS, byteDevicePointers[iteration], PIXEL_PER_IMAGE, iteration);

	gpuErrchk(cudaPeekAtLastError());

	gpuErrchk(cudaMemcpy(hp_destination.get(), byteDevicePointers[iteration], memory_size, cudaMemcpyDeviceToHost));

	auto const data{ bitmaps[iteration].pixel_span().data() };
	// copy to bmp
	for (auto i{ 0 }; i < PIXEL_PER_IMAGE; ++i) {
		data[i] = { 0, 0, hp_destination[i] };
	}
}

void calculateOnDeviceByte_parallel(int const iteration, int const offset) {
	static auto const memory_size{ memory_size_byte };

	auto hp_destination{ std::make_unique<pfc::byte_t[]>(memory_size) };

	call_mandel_kernel(BLOCKS, THREADS, byteDevicePointers[iteration - offset], PIXEL_PER_IMAGE, iteration);

	gpuErrchk(cudaPeekAtLastError());

	gpuErrchk(cudaMemcpy(hp_destination.get(), byteDevicePointers[iteration - offset], memory_size, cudaMemcpyDeviceToHost));

	auto const data{ bitmaps[iteration + offset].pixel_span().data() };
	// copy to bmp
	for (auto i{ 0 }; i < PIXEL_PER_IMAGE; ++i) {
		data[i] = { 0, 0, hp_destination[i] };
	}
}

// GPU functions
void sequential_gpu_byte(int const images) {
	static auto const memory_size{ memory_size_bgr };
	for (auto o{ 0 }; o < images; ++o)
	{
		call_mandel_kernel(BLOCKS, THREADS, bgrDevicePointers[0], PIXEL_PER_IMAGE, o);

		gpuErrchk(cudaPeekAtLastError());

		gpuErrchk(cudaMemcpy(bitmaps[o].pixel_span().data(), bgrDevicePointers[0], memory_size, cudaMemcpyDeviceToHost));
	}
}

void sequential_gpu_bitmap(int const images) {
	for (auto o{ 0 }; o < images; ++o)
	{
		calculateOnDeviceBitmap(o);
	}
}

void parallel_gpu_byte_all(int const images) {
	
	static int size;
	size = images / 2;
	pfc::parallel_range(true, size, size, [](int const o, int const begin, int const end) {
		calculateOnDeviceByte_parallel(o);
	});

	pfc::parallel_range(true, size, size, [](int const o, int const begin, int const end) {
		calculateOnDeviceByte_parallel(o + size, size);
	});
}

void parallel_gpu_bitmap_all(int const images) {
	
	pfc::parallel_range(true, CPU_PARALLEL_SIZE, images, [](int const o, int const begin, int const end) {
		calculateOnDeviceBitmap(o);
	});
}

void parallel_gpu_bitmap_chunked(int const images, int const chunk_size) {
	
	pfc::parallel_range(true, chunk_size, images, [](int const o, int const begin, int const end) {
		for (auto i{ begin }; i < end; ++i) {
			calculateOnDeviceBitmap(i);
		}
		});
}

void parallel_gpu_byte_chunked(int const images, int const chunk_size) {
	
	pfc::parallel_range(true, chunk_size, images, [](int const o, int const begin, int const end) {
		for (auto i{ begin }; i < end; ++i)
			calculateOnDeviceByte(i);
		});
}

void freeGPU() {
	gpuErrchk(cudaDeviceReset());
	std::cout << "Freed GPU memory" << std::endl;
}

cudaStream_t streams[CPU_PARALLEL_SIZE];

void initGPU() {
	std::cout << "Alloc GPU memory" << std::endl;

	for (auto i{ 0 }; i < deviceSize; ++i) {
		pfc::BGR_4_t* dp_destination_bgr{ nullptr };
		gpuErrchk(cudaMalloc(&dp_destination_bgr, memory_size_bgr));
		bgrDevicePointers.emplace_back(dp_destination_bgr);

		gpuErrchk(cudaStreamCreate(&streams[i]));
	}

	for (auto i{ 0 }; i < mappingSize; ++i) {
		pfc::byte_t* dp_destination_byte{ nullptr };

		gpuErrchk(cudaMalloc(&dp_destination_byte, memory_size_byte));

		byteDevicePointers.emplace_back(dp_destination_byte);
	}

	std::cout << "Allocated GPU memory" << std::endl;
}

auto constexpr no_of_images_per_stream{ 200 / CPU_PARALLEL_SIZE };
auto constexpr no_of_streams{ CPU_PARALLEL_SIZE };

void parallel_streamed_GPU(int const images) {
	static auto const memory_size{ memory_size_bgr };
	for (int t = 0; t < 10; ++t) {
		gpuErrchk(cudaStreamCreate(&streams[t]));
		call_mandel_kernel(BLOCKS, THREADS, bgrDevicePointers[t], PIXEL_PER_IMAGE, t, streams[t]);
		gpuErrchk(cudaGetLastError());

		//gpuErrchk(cudaMemcpy(bitmaps[t].pixel_span().data(), bgrDevicePointers[t], memory_size, cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpyAsync(bitmaps[t].pixel_span().data(), bgrDevicePointers[t], PIXEL_PER_IMAGE, cudaMemcpyDeviceToHost, streams[t]));
		gpuErrchk(cudaGetLastError());

		gpuErrchk(cudaStreamSynchronize(streams[t]));
	}

	//pfc::parallel_range(true, CPU_PARALLEL_SIZE, images, [](int t, int begin, int end) {
	//	auto const t_{ t + 1 };

	//	for (auto i{ 0 }; i < CPU_PARALLEL_SIZE; ++i) {
	//		auto const image{ 10 * i + t_ };

	//		call_mandel_kernel(BLOCKS, THREADS, bgrDevicePointers[t], PIXEL_PER_IMAGE, image, streams[t]);
	//		gpuErrchk(cudaPeekAtLastError());

	//		gpuErrchk(cudaMemcpyAsync(bitmaps[t].pixel_span().data(), bgrDevicePointers[t], PIXEL_PER_IMAGE, cudaMemcpyDeviceToHost, streams[t]));

	//		gpuErrchk(cudaStreamSynchronize(streams[t]));
	//	}
	//});
}
