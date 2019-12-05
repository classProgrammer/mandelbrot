#include <vector>
#include "mandelbrotutil.h"
#include "mandel.h"
#include "mandel_constants_gpu.h"
#include "pfc_threading.h"
#include<string>  

// TODO:
// CPU threads vs task with param settings when is what better
// try to be creative with paralelism on CPU, e.g. 4 parallel images which are calculated parallel
// c++ complex class => slow, but use
// NVIDIA complex class => slow, but use
// define infinity etc. 
// microsoft gsl
// types.h use on device 


void runTests() {
	auto const images{ NO_OF_IMAGES };
	auto const chunk_size_bitmap{ 50 }; 
	auto const BEST_INNER_SIZE{ WIDTH / 16 };

	// precalcualtes values to increase performance
	initGPU();
	initCPU();

	std::vector<float> times;
	std::vector<std::string> labels;

	// ---------------------------------------------------------------------------------
	// --------------------------------   GPU   ----------------------------------------
	// ---------------------------------------------------------------------------------

	//times.emplace_back(measureTime(parallel_streamed_GPU_prallel_range, 20));
	//labels.emplace_back("gpu_paralell_range_stream");
	//storeLastImage(20, "gpu_paralell_range_stream");

	//times.emplace_back(measureTime(parallel_streamed_GPU_for_loop, 20));
	//labels.emplace_back("gpu_for_streamed");
	//storeLastImage(20, "gpu_for_streamed");

	times.emplace_back(measureTime(sequential_gpu, 200));
	labels.emplace_back("gpu_sequential");
	storeLastImage(200, "gpu_sequential");

	//times.emplace_back(measureTime(parallel_GPU_stream0, 20));
	//labels.emplace_back("gpu_parallel_stream0");
	//storeLastImage(20, "gpu_parallel_stream0");

	printResult(std::cout, 20, times, labels);
	freeGPU();
}

int main() {

	if (!foundDevice()) {
		std::cout << "!!!!!!!!!! FAILED, no device found !!!!!!!!!!" << std::endl;
		return 1;
	}

	runTests();

	return 0;
}