all:
	nvcc -arch=sm_60 -Xptxas -dlcm=cg main.cpp histogram.cu -o histogram -O3 -L/usr/X11R6/lib -lm -lpthread -lX11

perf:
	nvcc -arch=sm_60 -Xptxas -dlcm=cg main.cpp histogram.cu -o histogram -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -g -lineinfo
