#include <random>
#include <thread>
#include <chrono>
#include "kernel.cuh"
#include "mandelbrotutil.h"
#include "mandel_constants_gpu.h"

void printBlock(int const no_of_images, float const data, float const mib, float const& time, std::string const& label) {
	std::cout <<
		"  +++++ " << label << " +++++" << std::endl <<
		"     Runtime:    " << time << " sec (for " << no_of_images << " bitmaps and " << data << " GiB)" << std::endl <<
		"     throughput: " << (mib * no_of_images) / time << " MiB/s"
		<< std::endl << std::endl;
}

void printFastest(int const no_of_images, float const data, float const mib, float const& time, std::string const& label) {
	std::cout <<
		"  ++++++++ FASTEST SOLUTION ++++++++" << std::endl <<
		"  +++++ " << label << " +++++" << std::endl <<
		"     Runtime:    " << time << " sec (for " << no_of_images << " bitmaps and " << data << " GiB)" << std::endl <<
		"     throughput: " << (mib * no_of_images) / time << " MiB/s"
		<< std::endl << std::endl;
}

void printHeader(float const& mib) {
	auto static constexpr CPU{ "Intel Core i7-7700K 4.20GHz" };
	auto static constexpr GPU{ "NVIDIA GTX 1080" };

	std::cout <<
		"++++++++++ RUN ++++++++++" << std::endl <<
		"     bitmap:     " << WIDTH << " x " << HEIGHT << " pels (" << mib << " MiB)" << std::endl <<
		"     CPU:        " << CPU << std::endl <<
		"     GPU:        " << GPU << std::endl << std::endl;
}

void printSpeedup(float const& time1, std::string const& label1, float const& time2, std::string const& label2) {
	std::cout <<
		"  +++++ " << "Speedup" << " +++++" << std::endl <<
		"     Speedup:    (" << label1 << "/" << label2 << "): " << (time1 / time2) << std::endl <<
		"     " << (time1 < time2 ? label1 : label2) << " is faster"
		<< std::endl << std::endl;
}

void printResult(int const no_of_images, std::vector<float> const& times, std::vector<std::string> const& labels) {
	// calculate elapsed times and speedup
	auto const mib{ ((float)WIDTH * HEIGHT * 4) / 1048576.0f };
	auto const gib{ ((float)WIDTH * HEIGHT * 4) / 1073741824.0f };
	auto const data{ no_of_images * gib };

	printHeader(mib);

	auto fastest{ 100000000.0f };
	auto fastest_idx{ 0 };

	for (auto i{ 0 }; i < times.size(); ++i) {
		printBlock(no_of_images, data, mib, times[i], labels[i]);
		if (times[i] < fastest) {
			fastest = times[i];
			fastest_idx = i;
		}
	}

	for (auto i{ 0 }; i < times.size(); ++i) {
		for (auto j{ i + 1}; j < times.size(); ++j) {
			printSpeedup(times[i], labels[i], times[j], labels[j]);	
		}
	}

	printFastest(no_of_images, data, mib, times[fastest_idx], labels[fastest_idx]);
	std::cout << "++++++++ END RUN ++++++++" << std::endl << std::endl;
}

bool foundDevice() {
	auto count{ 0 };
	gpuErrchk(cudaGetDeviceCount(&count));
	if (count < 1) return false;

	auto device_no{ 0 };
	gpuErrchk(cudaSetDevice(device_no));

	return true;
}