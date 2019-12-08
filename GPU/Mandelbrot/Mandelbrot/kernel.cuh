#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include<iostream>
#include "pfc_types.h"

void call_mandel_kernel(dim3 const& big, dim3 const& tib, pfc::BGR_4_t* dest, int const size, int const outer_idx);
void call_mandel_kernel(dim3 const& big, dim3 const& tib, pfc::BGR_4_t* dest, int const size, int const outer_idx, cudaStream_t &stream);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code == cudaSuccess) return;

	fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	if (abort) exit(code);
}