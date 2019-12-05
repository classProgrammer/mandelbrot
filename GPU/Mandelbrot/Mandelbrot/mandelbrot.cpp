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

	auto const chunk_size_bitmap{ 50 }; // 70 sometimes out of memory
	// best is 1/4 row
	auto const BEST_INNER_SIZE{ WIDTH / 16 };

	// precalcualtes values to increase performance
	initGPU();
	initCPU();
	//factorWrite();

	std::vector<float> times;
	std::vector<std::string> labels;

	//times.emplace_back(measureTime(global_parallel_local_parallel_task, images, BEST_INNER_SIZE, WORK_PER_VIRTUAL_CORE));
	//labels.emplace_back("GP_LP_1.4_task_virtual_v1");

	//times.emplace_back(measureTime(global_parallel_local_parallel_task2, images, BEST_INNER_SIZE, WORK_PER_VIRTUAL_CORE));
	//labels.emplace_back("GP_LP_1.4_task_virtual_v2");

	// --------------------------------------------------

	//times.emplace_back(measureTime(global_sequential_local_prallel_task, images, BEST_INNER_SIZE));
	//labels.emplace_back("GS_LP_1.4_task");

	//times.emplace_back(measureTime(global_parallel_local_parallel_task, images, BEST_INNER_SIZE, WORK_PER_VIRTUAL_CORE));
	//labels.emplace_back("GP_LP_1.4_task_virtual");

	//times.emplace_back(measureTime(global_parallel_local_parallel_task, images, WIDTH / 2, WORK_PER_VIRTUAL_CORE));
	//labels.emplace_back("GP_LP_W2_task_virtual");

	//times.emplace_back(measureTime(global_parallel_local_parallel_task, images, WIDTH, WORK_PER_VIRTUAL_CORE));
	//labels.emplace_back("GP_LP_W_task_virtual");

	//times.emplace_back(measureTime(global_parallel_local_parallel_task, images, WIDTH / 8, WORK_PER_VIRTUAL_CORE));
	//labels.emplace_back("GP_LP_1.8W_task_virtual");	

	//times.emplace_back(measureTime(global_parallel_local_parallel_task, images, WIDTH / 16, WORK_PER_VIRTUAL_CORE));
	//labels.emplace_back("GP_LP_1.16W_task_virtual");
	//
	//times.emplace_back(measureTime(global_parallel_local_parallel_task, images, WIDTH / 32, WORK_PER_VIRTUAL_CORE));
	//labels.emplace_back("GP_LP_1.32W_task_virtual");	
	//
	//times.emplace_back(measureTime(global_parallel_local_parallel_task, images, WIDTH / 16, 1));
	//labels.emplace_back("GP_LP_1.16W_task_all"); // BEST SO FAR

	//times.emplace_back(measureTime(global_parallel_local_parallel_task, images, WIDTH / 32, 1));
	//labels.emplace_back("GP_LP_1.32W_task_all");	

/*	times.emplace_back(measureTime(global_parallel_local_sequential_task, images, WORK_PER_VIRTUAL_CORE));
	labels.emplace_back("GP_LS_task_virtual")*/;

	// --------------------------------------------------

	//times.emplace_back(measureTime(global_parallel_local_parallel_task, images, BEST_INNER_SIZE, WORK_PER_PHYSICAL_CORE));
	//labels.emplace_back("GP_LP_1.4_task_physical");

	//times.emplace_back(measureTime(global_parallel_local_parallel_task, images, WIDTH / 2, WORK_PER_PHYSICAL_CORE));
	//labels.emplace_back("GP_LP_W2_task_physical");

	//times.emplace_back(measureTime(global_parallel_local_parallel_task, images, WIDTH, WORK_PER_PHYSICAL_CORE));
	//labels.emplace_back("GP_LP_W_task_physical");

	//times.emplace_back(measureTime(global_parallel_local_parallel_task, images, WIDTH / 8, WORK_PER_PHYSICAL_CORE));
	//labels.emplace_back("GP_LP_1.8W_task_physical");

	//times.emplace_back(measureTime(global_parallel_local_parallel_task, images, WORK_PER_PHYSICAL_CORE, images / 2));
	//labels.emplace_back("GP_LP_half_task_physical");

	//times.emplace_back(measureTime(global_parallel_local_sequential_task, images, WORK_PER_PHYSICAL_CORE));
	//labels.emplace_back("GP_LS_task_physical");

	// --------------------------------------------------

	//times.emplace_back(measureTime(global_parallel_local_parallel_task, images, BEST_INNER_SIZE, DUAL_CORE));
	//labels.emplace_back("GP_LP_1.4_task_dual");

	//times.emplace_back(measureTime(global_parallel_local_parallel_task, images, WIDTH / 2, DUAL_CORE));
	//labels.emplace_back("GP_LP_W2_task_dual");

	//times.emplace_back(measureTime(global_parallel_local_parallel_task, images, WIDTH, DUAL_CORE));
	//labels.emplace_back("GP_LP_W_task_dual");

	//times.emplace_back(measureTime(global_parallel_local_parallel_task, images, WIDTH / 8, DUAL_CORE));
	//labels.emplace_back("GP_LP_1.8W_task_dual");

	//times.emplace_back(measureTime(global_parallel_local_parallel_task, images, DUAL_CORE, images / 2));
	//labels.emplace_back("GP_LP_half_task_dual");

	//times.emplace_back(measureTime(global_parallel_local_sequential_task, images, DUAL_CORE));
	//labels.emplace_back("GP_LS_task_dual");

	//// --------------------------------------------------

	//times.emplace_back(measureTime(global_sequential_local_prallel_thread, images, BEST_INNER_SIZE));
	//labels.emplace_back("GS_LP_1.4_thread");

	//times.emplace_back(measureTime(global_parallel_local_parallel_thread, images, BEST_INNER_SIZE, WORK_PER_VIRTUAL_CORE));
	//labels.emplace_back("GP_LP_1.4_thread_virtual");

	//times.emplace_back(measureTime(global_parallel_local_parallel_thread, images, WIDTH / 2, WORK_PER_VIRTUAL_CORE));
	//labels.emplace_back("GP_LP_W2_thread_virtual");

	//times.emplace_back(measureTime(global_parallel_local_parallel_thread, images, WIDTH, WORK_PER_VIRTUAL_CORE));
	//labels.emplace_back("GP_LP_W_thread_virtual");

	//times.emplace_back(measureTime(global_parallel_local_parallel_thread, images, WIDTH / 8, WORK_PER_VIRTUAL_CORE));
	//labels.emplace_back("GP_LP_W1.8_thread_virtual");

	//times.emplace_back(measureTime(global_parallel_local_sequential_thread, images, WORK_PER_VIRTUAL_CORE));
	//labels.emplace_back("GP_LS_thread_virtual");

	// --------------------------------------------------

	//times.emplace_back(measureTime(sequential_gpu_byte, images));
	//labels.emplace_back("sequential_GPU_byte");	

	//times.emplace_back(measureTime(parallel_gpu_bitmap_all_opt, images));
	//labels.emplace_back("sequential_GPU_byte");

	//times.emplace_back(measureTime(global_parallel_local_parallel_task, images, BEST_INNER_SIZE));
	//labels.emplace_back("gp_lp_task_v1");


	//auto const no_of_tasks{ HEIGHT * images * 2 };
	//times.emplace_back(measureTime(global_parallel_local_parallel_task, images, WIDTH/ 16));
	//labels.emplace_back("GP_LP_task_v1");			
	//
	//times.emplace_back(measureTime(global_parallel_local_parallel_task2, 10, WIDTH/ 16));
	//labels.emplace_back("GP_LP_task_v2");		
	//storeImages(10, "GP_LP_task_v2");

	//times.emplace_back(measureTime(global_parallel_local_parallel_task3, images, WIDTH / 8));
	//labels.emplace_back("GP_LP_task_v3");		
	
	//times.emplace_back(measureTime(global_parallel_local_parallel_task2, images, WIDTH/ 16));
	//labels.emplace_back("GP_LP_task_");	


	//times.emplace_back(measureTime(sequential_gpu_bitmap, images));
	//labels.emplace_back("sequential_GPU_bitmap");
	//storeLastImage(200, "gpu");

	// ---------------------------------------------------------------------------------
	// --------------------------------   CPU   ----------------------------------------
	// ---------------------------------------------------------------------------------

	// GS_LS
	//times.emplace_back(measureTime(global_sequential_local_sequential, 10));
	//labels.emplace_back("gs_ls");
	//storeImages(10, "gs_ls");

	// GP_LS
	//times.emplace_back(measureTime(global_parallel_local_sequential_task, 40));
	//labels.emplace_back("gp_ls_task");
	//storeImages(40, "gp_ls_task");

	//times.emplace_back(measureTime(global_parallel_local_sequential_thread, 40));
	//labels.emplace_back("gp_ls_thread");
	//storeLastImage(40, "gp_ls_thread");

	// GS_LP
	//times.emplace_back(measureTime(global_sequential_local_prallel_task, 40, WIDTH / 16));
	//labels.emplace_back("gs_lp_task");
	//storeImages(40, "gs_lp_task");

	//times.emplace_back(measureTime(global_sequential_local_prallel_thread, 40, WIDTH / 16));
	//labels.emplace_back("gs_lp_task");
	//storeLastImage(40, "gs_lp_task");

	// GP_LP
	//times.emplace_back(measureTime(global_parallel_local_parallel_task, 200, WIDTH / 16));
	//labels.emplace_back("gp_lp_task");
	//storeImages(200, "gp_lp_task");

	
	//times.emplace_back(measureTime(global_parallel_local_parallel_thread, 40, WIDTH / 16));
	//labels.emplace_back("gp_lp_thread");
	//storeImages(40, "gp_lp_thread");

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