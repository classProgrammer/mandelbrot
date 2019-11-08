#pragma once

bool foundDevice();

// template to measure a functions execution time

template<typename F, typename... Args>
float measureTime(F& func, Args&&... args) {
	auto const device_start{ std::chrono::high_resolution_clock::now() };
	func(std::forward<Args>(args)...);
	auto const device_finish{ std::chrono::high_resolution_clock::now() };
	auto device_elapsed{ device_finish - device_start };
	typedef std::chrono::duration<float> float_seconds;
	
	return ((float)std::chrono::duration_cast<std::chrono::milliseconds>(device_elapsed).count()) / 1000.0f;
};

void printResult(float const& host_time, float const& device_time, std::string const& label1, std::string const& label2);
void printResult(float const& time1, float const& time2, std::string const& label1, std::string const& label2, int const no_of_images);