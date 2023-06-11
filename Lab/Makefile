all:
	nvcc -arch=sm_60 -Xptxas -dlcm=cg main.cpp scan.cu -o sort -O3 -L/usr/X11R6/lib -lm -lpthread -lX11
