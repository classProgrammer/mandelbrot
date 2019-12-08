#include "kernel.cuh"
#include "mandel.h"
#include "mandel_constants_gpu.h"

__global__ void mandel_kernel(pfc::BGR_4_t* dest, int const size, int const o) {
	auto const i{ blockIdx.x * blockDim.x + threadIdx.x };

	if (i < size) {
		dest[i].red = std::move(colorDevice((i % WIDTH) * X_FACTORS[o] + CX_MIN[o], (i / WIDTH) * Y_FACTORS[o] + CY_MIN[o]));
	}
}

void call_mandel_kernel(dim3 const& big, dim3 const& tib, pfc::BGR_4_t* dest, int const size, int const outer_idx) {
	mandel_kernel << <big, tib >> > (dest, size, outer_idx);
}

void call_mandel_kernel(dim3 const& big, dim3 const& tib, pfc::BGR_4_t* dest, int const size, int const outer_idx, cudaStream_t& stream) {
	mandel_kernel << <big, tib, 0, stream >> > (dest, size, outer_idx);
}
