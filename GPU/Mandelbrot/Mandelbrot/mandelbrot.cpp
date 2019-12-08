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

	initGPU();
	initCPU();

	std::vector<float> times;
	std::vector<std::string> labels;

	// ---------------------------------------------------------------------------------
	// --------------------------------   GPU   ----------------------------------------
	// ---------------------------------------------------------------------------------

	times.emplace_back(measureTime(parallel_streamed_GPU_prallel_range, images));
	labels.emplace_back("gpu_paralell_range_stream");
	storeLastImage(images, "gpu_paralell_range_stream");

	times.emplace_back(measureTime(parallel_streamed_GPU_for_loop, images));
	labels.emplace_back("gpu_for_streamed");
	storeLastImage(images, "gpu_for_streamed");

	times.emplace_back(measureTime(sequential_gpu, images));
	labels.emplace_back("gpu_sequential");
	storeLastImage(images, "gpu_sequential");

	printResult(std::cout, images, times, labels);
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