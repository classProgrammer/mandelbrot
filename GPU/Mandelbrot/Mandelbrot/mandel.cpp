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

pfc::bitmap bitmaps[CPU_PARALLEL_SIZE];
auto constexpr DEVICE_SIZE{ CPU_PARALLEL_SIZE };
auto constexpr MEMORY_SIZE{ PIXEL_PER_IMAGE * sizeof(pfc::BGR_4_t) };
std::vector<pfc::BGR_4_t*> bgrDevicePointers;
cudaStream_t streams[CPU_PARALLEL_SIZE];

void initCPU() {
	gpuErrchk(cudaSetDeviceFlags(cudaDeviceMapHost)); // needed for cudeGetDevicePointer to work
	std::cout << "Allocating Bitmaps" << std::endl;
	static constexpr auto max_images{ CPU_PARALLEL_SIZE };
	pfc::parallel_range(USE_TASKS, 10, max_images, [](int const o, int const begin, int const end) {
		for (auto i{ begin }; i < end; ++i) {
			bitmaps[i] = pfc::bitmap{ WIDTH, HEIGHT };
			auto data{ bitmaps[i].pixel_span().data() };
			for (auto i{ 0 }; i < (PIXEL_PER_IMAGE); ++i) {
				data[i] = { 0, 0, 0 };
			}
			gpuErrchk(cudaHostRegister(bitmaps[i].pixel_span().data(), MEMORY_SIZE, cudaHostRegisterMapped));
		}
		});
	std::cout << "Bitmaps Allocated" << std::endl;
}

void initGPU() {
	std::cout << "Alloc GPU memory" << std::endl;
	

	for (auto i{ 0 }; i < DEVICE_SIZE; ++i) {
		pfc::bitmap::pixel_type * dp_destination_bgr{ nullptr };

		// produces endless loop ???
		//gpuErrchk(cudaHostGetDevicePointer(&dp_destination_bgr, bitmaps[i].pixel_span().data(), 0))

		gpuErrchk(cudaMalloc(&dp_destination_bgr, MEMORY_SIZE));
		bgrDevicePointers.emplace_back(dp_destination_bgr);
		gpuErrchk(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
	}

	std::cout << "Allocated GPU memory" << std::endl;
}

void freeGPU() {
	gpuErrchk(cudaDeviceReset());
	std::cout << "Freed GPU memory" << std::endl;
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

void sequential_gpu(int const images) {
	
	for (auto o{ 0 }; o < images; ++o)
	{
		call_mandel_kernel(BLOCKS, THREADS, bgrDevicePointers[0], PIXEL_PER_IMAGE, o);

		gpuErrchk(cudaPeekAtLastError());

		gpuErrchk(cudaMemcpy(bitmaps[CPU_PARALLEL_SIZE - 1].pixel_span().data(), bgrDevicePointers[0], MEMORY_SIZE, cudaMemcpyDeviceToHost));
	}
}

void parallel_streamed_GPU_prallel_range(int const images) {

	pfc::parallel_range(USE_TASKS, DEVICE_SIZE, images, [images](int const t, int const begin, int const end) {

		for (auto i{ t }; i < images; i += CPU_PARALLEL_SIZE) {
			call_mandel_kernel(BLOCKS, THREADS, bgrDevicePointers[t], PIXEL_PER_IMAGE, i, streams[t]);
			gpuErrchk(cudaPeekAtLastError());

			gpuErrchk(cudaMemcpyAsync(bitmaps[t].pixel_span().data(), bgrDevicePointers[t], MEMORY_SIZE, cudaMemcpyDeviceToHost, streams[t]));

			gpuErrchk(cudaStreamSynchronize(streams[t]));
		}
	});
}