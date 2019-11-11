#pragma once
#include <vector>
#include <any>
#include <chrono>
#include <iostream>

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

void printResult(int const no_of_images, std::vector<float> const& times, std::vector<std::string> const& labels);