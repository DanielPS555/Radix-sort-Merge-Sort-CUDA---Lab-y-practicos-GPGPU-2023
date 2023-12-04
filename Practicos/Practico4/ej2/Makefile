all:
	nvcc -arch=sm_60 -Xptxas -dlcm=cg main.cpp blur.cu -o blur -O3 -L/usr/X11R6/lib -lm -lpthread -lX11
