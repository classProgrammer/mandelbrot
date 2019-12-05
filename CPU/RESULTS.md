# Mandelbrot Set

## Optimized Version of the Color Function
```c++
pfc::byte_t color(float const cr, float const ci) {
	auto iterations{ START_ITERATION };
	auto zr{ 0.0f },
		 zi{ 0.0f },
		 zr_sq{ 0.0f },
		 zi_sq{ 0.0f },
		 temp_i{ 0.0f };

	do {
		zr_sq = zr * zr;
		zi_sq = zi * zi;
		temp_i = zr * zi;
		zi = temp_i + temp_i + ci;
		zr = zr_sq - zi_sq + cr;

	} while (zr_sq + zi_sq < BORDER && --iterations);

	return COLORS[iterations];
}
```

## Best CPU vs GPU on University PC
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
The fastest version is of course the GPU version with a speedup of approximately 77 and a runtime of 7.858 seconds.

## CPU Version Comparison on 20 Images for Threads and Tasks
GS_LS is by far the slowest. Task are usually faster than threads and the fastest solution is GP_LP.
```text
++++++++++ RUN ++++++++++
     bitmap:     8192 x 4608 pels (144 MiB)
     CPU:        Intel Core i7-7700K 4.20GHz
     GPU:        NVIDIA GTX 1080

  +++++ gs_ls +++++
     Runtime:    133.44 sec (for 20 bitmaps and 2.8125 GiB)
     throughput: 21.5827 MiB/s

  +++++ gp_ls_task +++++
     Runtime:    22.822 sec (for 20 bitmaps and 2.8125 GiB)
     throughput: 126.194 MiB/s

  +++++ gp_ls_thread +++++
     Runtime:    19.531 sec (for 20 bitmaps and 2.8125 GiB)
     throughput: 147.458 MiB/s

  +++++ gs_lp_task +++++
     Runtime:    18.904 sec (for 20 bitmaps and 2.8125 GiB)
     throughput: 152.349 MiB/s

  +++++ gs_lp_thread +++++
     Runtime:    18.825 sec (for 20 bitmaps and 2.8125 GiB)
     throughput: 152.988 MiB/s

  +++++ gp_lp_task +++++
     Runtime:    18.465 sec (for 20 bitmaps and 2.8125 GiB)
     throughput: 155.971 MiB/s

  +++++ gp_lp_thread +++++
     Runtime:    19.407 sec (for 20 bitmaps and 2.8125 GiB)
     throughput: 148.4 MiB/s

  +++++ Speedup +++++
     Speedup:    (gs_ls/gp_ls_task): 5.84699
     gp_ls_task is faster

  +++++ Speedup +++++
     Speedup:    (gs_ls/gp_ls_thread): 6.83222
     gp_ls_thread is faster

  +++++ Speedup +++++
     Speedup:    (gs_ls/gs_lp_task): 7.05882
     gs_lp_task is faster

  +++++ Speedup +++++
     Speedup:    (gs_ls/gs_lp_thread): 7.08845
     gs_lp_thread is faster

  +++++ Speedup +++++
     Speedup:    (gs_ls/gp_lp_task): 7.22664
     gp_lp_task is faster

  +++++ Speedup +++++
     Speedup:    (gs_ls/gp_lp_thread): 6.87587
     gp_lp_thread is faster

  +++++ Speedup +++++
     Speedup:    (gp_ls_task/gp_ls_thread): 1.1685
     gp_ls_thread is faster

  +++++ Speedup +++++
     Speedup:    (gp_ls_task/gs_lp_task): 1.20726
     gs_lp_task is faster

  +++++ Speedup +++++
     Speedup:    (gp_ls_task/gs_lp_thread): 1.21232
     gs_lp_thread is faster

  +++++ Speedup +++++
     Speedup:    (gp_ls_task/gp_lp_task): 1.23596
     gp_lp_task is faster

  +++++ Speedup +++++
     Speedup:    (gp_ls_task/gp_lp_thread): 1.17597
     gp_lp_thread is faster

  +++++ Speedup +++++
     Speedup:    (gp_ls_thread/gs_lp_task): 1.03317
     gs_lp_task is faster

  +++++ Speedup +++++
     Speedup:    (gp_ls_thread/gs_lp_thread): 1.0375
     gs_lp_thread is faster

  +++++ Speedup +++++
     Speedup:    (gp_ls_thread/gp_lp_task): 1.05773
     gp_lp_task is faster

  +++++ Speedup +++++
     Speedup:    (gp_ls_thread/gp_lp_thread): 1.00639
     gp_lp_thread is faster

  +++++ Speedup +++++
     Speedup:    (gs_lp_task/gs_lp_thread): 1.0042
     gs_lp_thread is faster

  +++++ Speedup +++++
     Speedup:    (gs_lp_task/gp_lp_task): 1.02377
     gp_lp_task is faster

  +++++ Speedup +++++
     Speedup:    (gs_lp_task/gp_lp_thread): 0.974082
     gs_lp_task is faster

  +++++ Speedup +++++
     Speedup:    (gs_lp_thread/gp_lp_task): 1.0195
     gp_lp_task is faster

  +++++ Speedup +++++
     Speedup:    (gs_lp_thread/gp_lp_thread): 0.970011
     gs_lp_thread is faster

  +++++ Speedup +++++
     Speedup:    (gp_lp_task/gp_lp_thread): 0.951461
     gp_lp_task is faster

  ++++++++ FASTEST SOLUTION ++++++++
  +++++ gp_lp_task +++++
     Runtime:    18.465 sec (for 20 bitmaps and 2.8125 GiB)
     throughput: 155.971 MiB/s

++++++++ END RUN ++++++++
```