#pragma once

void init_CPU();
void global_sequential_local_prallel(int const images, int const inner_size);
void global_parallel_local_parallel(int const images, int const inner_size, int const outer_size);
void global_parallel_local_sequential(int const images, int const outer_size);
void global_sequential_local_sequential(int const images);