# Mandelbrot Set GPU Tuning

## Base Case: Result After Part 1
```txt
++++++++++ RUN ++++++++++
     bitmap:     8192 x 4608 pels (144 MiB)
     CPU:        Labor Rechner
     GPU:        NVIDIA 1070 GTX

  +++++ gp_lp_task +++++
     Runtime:    609.717 sec (for 200 bitmaps and 28.125 GiB)
     throughput: 47.235 MiB/s

  +++++ gpu_sequential +++++
     Runtime:    7.858 sec (for 200 bitmaps and 28.125 GiB)
     throughput: 3665.05 MiB/s

  +++++ Speedup +++++
     Speedup:    (gp_lp_task/gpu_sequential): 77.5919
     gpu_sequential is faster

  ++++++++ FASTEST SOLUTION ++++++++
  +++++ gpu_sequential +++++
     Runtime:    7.858 sec (for 200 bitmaps and 28.125 GiB)
     throughput: 3665.05 MiB/s

++++++++ END RUN ++++++++
```
The sequential GPU solution without any tuning needs ~7.9 seconds to complete. 

## Result on University Comnputer: Tuned GPU
The tuning approach is based on streams and asynchrounous memory copying with pinned memory.
```txt
++++++++++ RUN ++++++++++
     bitmap:     8192 x 4608 pels (144 MiB)
     CPU:        Labor Rechner
     GPU:        NVIDIA GTX 1070

  +++++ gpu_paralell_range_stream +++++
     Runtime:    2.674 sec (for 200 bitmaps and 28.125 GiB)
     throughput: 10770.4 MiB/s

  +++++ gpu_sequential +++++
     Runtime:    5.088 sec (for 200 bitmaps and 28.125 GiB)
     throughput: 5660.38 MiB/s

  +++++ Speedup +++++
     Speedup:    (gpu_paralell_range_stream/gpu_sequential): 0.52555
     gpu_paralell_range_stream is faster

  ++++++++ FASTEST SOLUTION ++++++++
  +++++ gpu_paralell_range_stream +++++
     Runtime:    2.674 sec (for 200 bitmaps and 28.125 GiB)
     throughput: 10770.4 MiB/s

++++++++ END RUN ++++++++

++++++++++ RUN ++++++++++
     bitmap:     8192 x 4608 pels (144 MiB)
     CPU:        Labor Rechner
     GPU:        NVIDIA GTX 1070

  +++++ gpu_paralell_range_stream +++++
     Runtime:    2.562 sec (for 200 bitmaps and 28.125 GiB)
     throughput: 11241.2 MiB/s

  +++++ gpu_sequential +++++
     Runtime:    4.991 sec (for 200 bitmaps and 28.125 GiB)
     throughput: 5770.39 MiB/s

  +++++ Speedup +++++
     Speedup:    (gpu_paralell_range_stream/gpu_sequential): 0.513324
     gpu_paralell_range_stream is faster

  ++++++++ FASTEST SOLUTION ++++++++
  +++++ gpu_paralell_range_stream +++++
     Runtime:    2.562 sec (for 200 bitmaps and 28.125 GiB)
     throughput: 11241.2 MiB/s

++++++++ END RUN ++++++++

++++++++++ RUN ++++++++++
     bitmap:     8192 x 4608 pels (144 MiB)
     CPU:        Labor Rechner
     GPU:        NVIDIA GTX 1070

  +++++ gpu_paralell_range_stream +++++
     Runtime:    2.599 sec (for 200 bitmaps and 28.125 GiB)
     throughput: 11081.2 MiB/s

  +++++ gpu_sequential +++++
     Runtime:    4.922 sec (for 200 bitmaps and 28.125 GiB)
     throughput: 5851.28 MiB/s

  +++++ Speedup +++++
     Speedup:    (gpu_paralell_range_stream/gpu_sequential): 0.528037
     gpu_paralell_range_stream is faster

  ++++++++ FASTEST SOLUTION ++++++++
  +++++ gpu_paralell_range_stream +++++
     Runtime:    2.599 sec (for 200 bitmaps and 28.125 GiB)
     throughput: 11081.2 MiB/s

++++++++ END RUN ++++++++
```
The three runs result in nearly the same time.
The regular sequential approach has been improved from ~7.9 seconds to ~4.9 seconds just by pinning the memory.
The streams with async memcopy have a big impact improving the time to ~2.6 seconds.
The final speedup on the GPU is ~3 = 7.9 / 2.6.
The sequential speedup is ~1.6 = 1.7 / 4.9.
The final speedup from GPU vs CPU is ~234 = 609 / 2.6.

## Approach
At first it looked like the perfect idea to use streams and async memcopy because obvious it should bring a huge speedup.
```text
++++++++++ RUN ++++++++++
     bitmap:     8192 x 4608 pels (144 MiB)
     CPU:        Intel Core i7-7700K 4.20GHz
     GPU:        NVIDIA GTX 1080

  +++++ gpu_paralell_range_stream +++++
     Runtime:    10.077 sec (for 200 bitmaps and 28.125 GiB)
     throughput: 2857.99 MiB/s

  +++++ gpu_sequential +++++
     Runtime:    10.003 sec (for 200 bitmaps and 28.125 GiB)
     throughput: 2879.14 MiB/s

  +++++ Speedup +++++
     Speedup:    (gpu_paralell_range_stream/gpu_sequential): 1.0074
     gpu_sequential is faster

  ++++++++ FASTEST SOLUTION ++++++++
  +++++ gpu_sequential +++++
     Runtime:    10.003 sec (for 200 bitmaps and 28.125 GiB)
     throughput: 2879.14 MiB/s

++++++++ END RUN ++++++++
```
It was very surprising that the speedup was zero although the memory is copied async and streams were used.
It made no sense at first. Here the code snippet which has not been modified to achieve the 2.6 seconds final time.
```cpp
void parallel_streamed_GPU_prallel_range(int const images) {

	pfc::parallel_range(USE_TASKS, DEVICE_SIZE, images, [images](int const t, int const begin, int const end) {

		for (auto i{ t }; i < images; i += CPU_PARALLEL_SIZE) {
			call_mandel_kernel(BLOCKS, THREADS, bgrDevicePointers[t], PIXEL_PER_IMAGE, i, streams[t]);
			gpuErrchk(cudaPeekAtLastError());

			gpuErrchk(cudaMemcpyAsync(bitmaps[t].pixel_span().data(), bgrDevicePointers[t], MEMORY_SIZE, cudaMemcpyDeviceToHost, streams[t]));

			gpuErrchk(cudaStreamSynchronize(streams[t]));
		}
	});
}
```
The code of the snippet already was perfect and when reading through the Cuda documentation and online examples the term "Pinned Memory" appeared. When reading through the documentation the functions "cudaHostMalloc" and "cudaHostRegister" could be identified as the solution to the previous problem.

Part of the description of "CudaHostAlloc":
```text
The driver tracks the virtual memory ranges allocated with this function and automatically accelerates calls to functions such as cudaMemcpy(). Since the memory can be accessed directly by the device, it can be read or written with much higher bandwidth than pageable memory obtained with functions such as malloc(). 
```
Simplified, pinning the memory has a huge impact on the speedup.
```cpp
gpuErrchk(cudaHostRegister(bitmaps[i].pixel_span().data(), MEMORY_SIZE, cudaHostRegisterMapped));
```
This one liner solved all the problems that where there before because now "cudaMemCopyAsync" and streams showed a huge impact.
Other people online statet that this one liner delivers a speedup of up to 2. Which is a lot for just one line of code. The downside is that pinned memory is a scarse resource and should not be overused because overusage leeds to a decrease of runtime.
To test the base case I tried one stream vs the sequential version which should again deliver about the same time.
```text
++++++++++ RUN ++++++++++
     bitmap:     8192 x 4608 pels (144 MiB)
     CPU:        Intel Core i7-7700K 4.20GHz
     GPU:        NVIDIA GTX 1080

  +++++ gpu_paralell_range_stream +++++
     Runtime:    6.474 sec (for 200 bitmaps and 28.125 GiB)
     throughput: 4448.56 MiB/s

  +++++ gpu_sequential +++++
     Runtime:    6.494 sec (for 200 bitmaps and 28.125 GiB)
     throughput: 4434.86 MiB/s

  +++++ Speedup +++++
     Speedup:    (gpu_paralell_range_stream/gpu_sequential): 0.99692
     gpu_paralell_range_stream is faster

  ++++++++ FASTEST SOLUTION ++++++++
  +++++ gpu_paralell_range_stream +++++
     Runtime:    6.474 sec (for 200 bitmaps and 28.125 GiB)
     throughput: 4448.56 MiB/s

++++++++ END RUN ++++++++
```
Which is the case here (Times are taken on my PC not the unversity PC here).
This shows that the stream approach in the case of the mandelbrot set only works with pinned memory in this case.

## Monitoring
I DID NOT MANAGE TO START THE MONITOR ON THE UNIVERSITY COMPUTER HENCE ALL THE RESULTS SHOWN ARE FROM MY PC AT HOME.

### Sequential GPU usage no tuning:
![seq_gpu_usage](./seq_gpu_usage.png)
### Sequential GPU usage tuned:
![seq_gpu_usage_pinned](./seq_gpu_pinned.png)
### Parallel Streamed GPU usage:
![parallel_streamed_usage](./par_stream.png)

These result need to be taken with a grain of salt as the detailed analysis schows a different result.

## Detailed Parallel Streamed Result
![cpu_usage](./cpu_usage.png)

Here it can be clearly seen that most of the time actual work is in progress the CPUs cores are used at max capacity and the value above therefore shows the avg usage.

When looking at the streams the following result is produced.
![stream_usage](./streams.png)

At my home PC only 10 streams work in parallel. That might be because the memory of the graphics card is used at 100% while the mandelbrot is processed at all times and results in the avg usage of 66.2% from above because it is used at 0% at the start where the bitmaps and streams are allocated.

When the values of the  streams are summed up an avg usage of 42% is the result far more than the 27.8% from the summary above.

## Detailed Sequential Tuned Result
![seq_cpu_usage](./seq_cpu_usage.png)

The sequential version makes less use of the CPU.

![seq_gpu_usage_detailed](./seq_gpu_usg.png)

Here it can be seen that one stream works all the time hence the behaviour is as expected.

## Detailed Result Sequential No Tuning
![seq_not_tuned_result](./seq_not_tuned_result.png)

It looks kind of the same when compared to the tuned version with the difference that it takes less time and that the GPU is used with 18% instead of the 29% from the tuned version.

## Detailed Result Parallel Streamed Unpinned Memory
![parallel_unpinned_result](./parallel_unpinned_result.png)

It can be clearly seen that the streams are not working parallel but sequential when the memory is not pinned.
This results in a longer runtime and also in inefficient use of the CPU.