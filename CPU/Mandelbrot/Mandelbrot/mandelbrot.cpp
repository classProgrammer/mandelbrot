#include <vector>
#include "mandelbrotutil.h"
#include "mandel_cpu.h"
#include "mandel_gpu.h"
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
	auto const images{ 25 };

	auto const chunk_size_bitmap{ 50 }; // 70 sometimes out of memory
	// best is 1/4 row
	auto const BEST_INNER_SIZE{ WIDTH / 16 };


	// precalcualtes values to increase performance
	init_CPU();

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

	//storeImages(95, 100, "gp_lp_v1");


	
	auto const no_of_tasks{ HEIGHT * images * 2 };
	times.emplace_back(measureTime(global_parallel_local_parallel_task3, images, no_of_tasks));
	labels.emplace_back("GP_LP_task_v3_2");	

	//storeImages(185, 200, "gp_lp_v3");

	printResult(std::cout, images, times, labels);

}

int main() {

	if (!foundDevice()) {
		std::cout << "!!!!!!!!!! FAILED, no device found !!!!!!!!!!" << std::endl;
	}




	//unsigned int size{ 10 };
	//unsigned int innerSize{ 5 };
	//unsigned int no_of_imges{ 3 };

	//for (int i = 0; i < size * no_of_imges; i += innerSize) {
	//	auto begin = i;
	//	auto end = i + innerSize;

	//	auto const o{ begin / size };
	//	begin %= size;
	//	end = (end -1) % size + 1;
	//	
	//	std::cout << "range (" << begin << " - " << end << "), image: " << o << std::endl;
	//}



	//const unsigned long int images = 3;
	//const unsigned long  pixel = 20;
	//const unsigned int height = 5;
	//const unsigned int width = 4;
	//long long int const size{ pixel * images };
	//const int inner_size = height * images;
	//// height * images = one task per WIDTH
	//// height * images * 2 = one task per WIDTH / 2

	//static auto const one_unit = size / inner_size;
	//static auto const threads_per_img = inner_size / images;

	//std::cout << "inner_size = " << inner_size << ", one_unit = " << one_unit << std::endl;

	//pfc::parallel_range(true, inner_size, inner_size, [size, inner_size](int thread_idx, unsigned int begin, unsigned int end) {

	//	//o = ?

	//	//int const o{ begin / size };
	//	begin = thread_idx % threads_per_img * one_unit ;
	//	end = begin + one_unit;
	//	auto const image = thread_idx / threads_per_img;

	//	std::string str = "image " + std::to_string(image) + " => (" + std::to_string(begin) + ", " + std::to_string(end)+ ")\n";
	//	
	//	std::cout << str;

	//	});

	runTests();
	return 0;
}