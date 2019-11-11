#pragma once
// GPU functions
void sequential_gpu_byte(int const images);

void sequential_gpu_bitmap(int const images);

void parallel_gpu_byte_all(int const images);

void parallel_gpu_byte_all_opt(int const images);

void parallel_gpu_bitmap_all(int const images);

void parallel_gpu_bitmap_chunked(int const images, int const chunk_size);

void parallel_gpu_byte_chunked(int const images, int const chunk_size);