#pragma once

#include <iostream>

void init_CPU();
void global_sequential_local_prallel_task(int const images, int const inner_size);
void global_parallel_local_parallel_task(int const images, int const inner_size, int const outer_size);
void global_parallel_local_sequential_task(int const images, int const outer_size);

void global_parallel_local_parallel_task2(int const images, int const inner_size);

void global_sequential_local_prallel_thread(int const images, int const inner_size);
void global_parallel_local_parallel_thread(int const images, int const inner_size, int const outer_size);
void global_parallel_local_sequential_thread(int const images, int const outer_size);


void global_sequential_local_sequential(int const images);
void storeImagesCPU(int const images, std::string const& prefix);