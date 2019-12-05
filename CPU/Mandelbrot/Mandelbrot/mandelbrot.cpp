#include <vector>
#include "mandelbrotutil.h"
#include "mandel.h"
#include "mandel_constants_gpu.h"
#include "pfc_threading.h"
#include<string>  

void runTests() {
	auto const images{ NO_OF_IMAGES };
	auto const BEST_INNER_SIZE{ WIDTH / 16 };

	initGPU();
	initCPU();

	std::vector<float> times;
	std::vector<std::string> labels;

	// ---------------------------------------------------------------------------------
	// --------------------------------   CPU   ----------------------------------------
	// ---------------------------------------------------------------------------------

	// GS_LS
	times.emplace_back(measureTime(global_sequential_local_sequential, images));
	labels.emplace_back("gs_ls");
	storeLastImage(images, "gs_ls");

	// GP_LS
	times.emplace_back(measureTime(global_parallel_local_sequential_task, images));
	labels.emplace_back("gp_ls_task");
	storeLastImage(images, "gp_ls_task");

	times.emplace_back(measureTime(global_parallel_local_sequential_thread, images));
	labels.emplace_back("gp_ls_thread");
	storeLastImage(images, "gp_ls_thread");

	// GS_LP
	times.emplace_back(measureTime(global_sequential_local_prallel_task, images, BEST_INNER_SIZE));
	labels.emplace_back("gs_lp_task");
	storeLastImage(images, "gs_lp_task");

	times.emplace_back(measureTime(global_sequential_local_prallel_thread, images, BEST_INNER_SIZE));
	labels.emplace_back("gs_lp_thread");
	storeLastImage(images, "gs_lp_thread");

	// GP_LP
	times.emplace_back(measureTime(global_parallel_local_parallel_task, images, BEST_INNER_SIZE));
	labels.emplace_back("gp_lp_task");
	storeLastImage(images, "gp_lp_task");

	
	times.emplace_back(measureTime(global_parallel_local_parallel_thread, images, BEST_INNER_SIZE));
	labels.emplace_back("gp_lp_thread");
	storeLastImage(images, "gp_lp_thread");

	// ---------------------------------------------------------------------------------
	// --------------------------------   GPU   ----------------------------------------
	// ---------------------------------------------------------------------------------

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