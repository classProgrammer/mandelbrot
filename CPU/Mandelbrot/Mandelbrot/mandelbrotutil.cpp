#include <random>
#include <thread>
#include <chrono>
#include "kernel.cuh"
#include "mandelbrot.h"

using pointCollection = std::unique_ptr<float3[]>;
using resultCollection = std::unique_ptr<int[]>;
using timePoint = std::chrono::steady_clock::time_point;

auto constexpr sizeofInt{ sizeof(int) };
auto constexpr sizeofFloat3{ sizeof(float3) };

// prints the result times and speedup for host and device
void printResult(float const& time1, float const& time2, std::string const& label1, std::string const& label2) {
	// calculate elapsed times and speedup
	std::cout << label1 + " time (" << time1 << " ms) : " + label2 + " time(" << time2 << " ms)" << std::endl
		<< "speedup: " << time1 / time2 << std::endl << (time1 < time2 ? label1 : label2) << " faster"
		<< std::endl << std::endl;
}

auto constexpr CPU{ "Intel Core i7-7700K 4.20GHz" };
auto constexpr GPU{ "NVIDIA GTX 1080" };

// prints the result times and speedup for host and device
void printResult(float const& time1, float const& time2, std::string const& label1, std::string const& label2, int const no_of_images) {
	// calculate elapsed times and speedup
	auto const mib{ ((float)WIDTH * HEIGHT * 4) / 1048576.0f };
	auto const gib{ ((float)WIDTH * HEIGHT * 4) / 1073741824.0f };
	auto const data{ no_of_images * gib };

	std::cout <<
		"++++++++++ RUN ++++++++++" << std::endl <<
		"     bitmap:     " << WIDTH << " x " << HEIGHT << " pels (" << mib << " MiB)" << std::endl <<
		"     CPU:        " << CPU << std::endl <<
		"     GPU:        " << GPU << std::endl << std::endl <<
		"  +++++ " << label1 << " +++++" << std::endl <<
		"     Runtime:    " << time1 << " sec (for " << no_of_images << " bitmaps and " << data << " GiB)" << std::endl <<
		"     throughput: " << (mib * no_of_images) / time1 << " MiB/s"
		<< std::endl << std::endl <<
		"  +++++ " << label2 << " +++++" << std::endl <<
		"     Runtime:    " << time2 << " sec (for " << no_of_images << " bitmaps and " << data << " GiB)" << std::endl <<
		"     throughput: " << (mib * no_of_images) / time2 << " MiB/s"
		<< std::endl << std::endl <<
		"  +++++ " << "Speedup" << " +++++" << std::endl <<
		"     Speedup:    (" << label1 << "/" << label2 << "): " << (time1 / time2) << std::endl <<
		"     " << (time1 < time2 ? label1 : label2) << " is faster"
		<< std::endl <<
		"++++++++ END RUN ++++++++" << std::endl << std::endl;
}

bool foundDevice() {
	auto count{ 0 };
	gpuErrchk(cudaGetDeviceCount(&count));
	if (count < 1) return false;

	auto device_no{ 0 };
	gpuErrchk(cudaSetDevice(device_no));

	return true;
}