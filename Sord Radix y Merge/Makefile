all:
	nvcc -arch=sm_60 -Xptxas -dlcm=cg  main.cpp lab.cu -o sort -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -std=c++11
	nvcc -arch=sm_60 -Xptxas -dlcm=cg  performanceTest.cpp lab.cu -o performaceTest -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -std=c++11
