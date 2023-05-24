#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "math.h"
#include <algorithm>

#include "include/histogram.h"

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define MS(f,elap)                                                                                           \
        double elap=0;                                                                                       \
        {                                                                                                    \
        struct timespec t_ini,t_fin;                                                                         \
            clock_gettime(CLOCK_MONOTONIC, &t_ini);                                                          \
            f;                                                                                               \
            clock_gettime(CLOCK_MONOTONIC, &t_fin);                                                          \
            elap = 1000 * (t_fin.tv_sec - t_ini.tv_sec) + (t_fin.tv_nsec - t_ini.tv_nsec)/1000000.0;         \
        }

// Kernels
__global__ void simple_histogram_kernel(float *img_gpu_in, float *img_gpu_out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float color = COLOR_SIZE; // Out of range
    if (x < width && y < height) {
        color = img_gpu_in[x + y * width];
    }

    if (color < (float)COLOR_SIZE) {
        atomicAdd(&img_gpu_out[(int)color], 1.f);
    }
}

__global__ void shared_memory_histogram_kernel(float *img_gpu_in, float *img_gpu_out, int width, int height) {
    __shared__ float h_block[COLOR_SIZE];

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    if (tid < COLOR_SIZE) {
        h_block[tid] = 0.f;
    }

    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float color = COLOR_SIZE; // Out of range
    if (x < width && y < height) {
        color = img_gpu_in[x + y * width];
    }

    if (color < (float)COLOR_SIZE) {
        atomicAdd(&h_block[(int)color], 1.f);
    }

    __syncthreads();

    if (tid < COLOR_SIZE) {
        atomicAdd(&img_gpu_out[tid], h_block[tid]);
    }
}


// Kernel callers
void gpu_execute_kernel(algorithm_type algo, const dim3 &gridSize, const dim3 &blockSize, float *img_gpu_in, float *img_gpu_out, int width, int height) {
    switch (algo) {
        case SIMPLE_HISTOGRAM:
            simple_histogram_kernel<<<gridSize, blockSize>>>(img_gpu_in, img_gpu_out, width, height);
            break;
        case SHARED_MEMORY_HISTOGRAM:
            shared_memory_histogram_kernel<<<gridSize, blockSize>>>(img_gpu_in, img_gpu_out, width, height);
            break;
        case IMPROVED_SHARED_MEMORY_HISTOGRAM:
            // improved_transpose_dummy_kernel<<<gridSize, blockSize>>>(img_gpu_in, img_gpu_out, width, height);
            break;
    }
    CUDA_CHK(cudaGetLastError())
    CUDA_CHK(cudaDeviceSynchronize())
}

// Utils
void allocate_and_copy_gpu(float* &gpu_in, float* &gpu_out, float *cpu_in, float *cpu_out, int width, int height) {
    size_t size = width * height * sizeof(float);
    size_t hist_size = COLOR_SIZE * sizeof(float);

    CUDA_CHK ( cudaMalloc((void**)& gpu_in, size) )

    // Initialize gpu_out in 0
    CUDA_CHK ( cudaMalloc((void**)& gpu_out, hist_size) )

    CUDA_CHK ( cudaMemcpy(gpu_in, cpu_in, size, cudaMemcpyHostToDevice) )
    CUDA_CHK ( cudaMemcpy(gpu_out, cpu_out, hist_size, cudaMemcpyHostToDevice) )

    CUDA_CHK ( cudaMemset(gpu_out, 0, hist_size) )
}

void copy_and_free_gpu(float* &gpu_in, float* &gpu_out, float *cpu_out, int width, int height) {
    size_t hist_size = COLOR_SIZE * sizeof(float);
    CUDA_CHK ( cudaMemcpy(cpu_out, gpu_out, hist_size, cudaMemcpyDeviceToHost) )
    CUDA_CHK ( cudaFree(gpu_in) )
    CUDA_CHK ( cudaFree(gpu_out) )
}

double execute_kernel(algorithm_type algo, float* in_cpu_m, float* out_cpu_m, int width, int height) {
    float * img_gpu = NULL, * img_gpu_out = NULL;
    allocate_and_copy_gpu(img_gpu, img_gpu_out, in_cpu_m, out_cpu_m, width, height);

    // TODO: Assume the image is multiple of BLOCK_SIZE
    dim3 gridSize( (int)((float)width)/BLOCK_SIZE, (int)((float)height)/BLOCK_SIZE ); // 40 x 30
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    MS(gpu_execute_kernel(algo, gridSize, blockSize, img_gpu, img_gpu_out, width, height), time)

    copy_and_free_gpu(img_gpu, img_gpu_out, out_cpu_m, width, height);

    return time;
}
