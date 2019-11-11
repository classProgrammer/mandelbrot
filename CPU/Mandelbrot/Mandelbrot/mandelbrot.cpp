#include <vector>
#include "mandelbrotutil.h"
#include "mandel_cpu.h"
#include "mandel_gpu.h"
#include "mandel_constants_gpu.h"

// TODO:
// CPU threads vs task with param settings when is what better
// try to be creative with paralelism on CPU, e.g. 4 parallel images which are calculated parallel
// c++ complex class => slow, but use
// NVIDIA complex class => slow, but use
// define infinity etc. 
// microsoft gsl
// types.h use on device 


int main() {

	if (!foundDevice()) {
		std::cout << "!!!!!!!!!! FAILED, no device found !!!!!!!!!!" << std::endl;
	}

	auto const images{ 200 };
	auto const cores{ 8 };
	auto const WORK_PER_CORE{ images / cores };
	auto const chunk_size_bitmap{ 50 }; // 70 sometimes out of memory

	// best is 1/4 row
	auto const BEST_INNER_SIZE{ WIDTH / 4 };

	// precalcualtes values to increase performance
	init_CPU();

	//factorWrite();

	std::vector<float> times;
	std::vector<std::string> labels;

	times.emplace_back(measureTime(global_sequential_local_prallel, images, BEST_INNER_SIZE));
	labels.emplace_back("GS_LP_1.4");

	times.emplace_back(measureTime(global_parallel_local_parallel, images, BEST_INNER_SIZE, WORK_PER_CORE));
	labels.emplace_back("GP_LP_1.4");

	times.emplace_back(measureTime(global_parallel_local_parallel, images, WIDTH / 2, WORK_PER_CORE));
	labels.emplace_back("GP_LP_W2");

	times.emplace_back(measureTime(global_parallel_local_parallel, images, WIDTH, WORK_PER_CORE));
	labels.emplace_back("GP_LP_W");

	times.emplace_back(measureTime(global_parallel_local_parallel, images, BEST_INNER_SIZE, images / 2));
	labels.emplace_back("GP_LP_half");

	times.emplace_back(measureTime(global_parallel_local_sequential, images, WORK_PER_CORE));
	labels.emplace_back("GP_LS");	
	
	times.emplace_back(measureTime(parallel_gpu_byte_all_opt, images));
	labels.emplace_back("GP_LS");

	/*times.emplace_back(measureTime(global_sequential_local_sequential, images));
	labels.emplace_back("GS_LS");*/

	printResult(images, times, labels);

	return 0;
}