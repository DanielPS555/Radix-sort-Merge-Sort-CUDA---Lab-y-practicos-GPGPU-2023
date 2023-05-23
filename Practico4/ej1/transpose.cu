#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "math.h"
#include <algorithm>

#include "include/transpose.h"

#define BLOCK_SIZE 32

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

/**
 * Simple transpose kernel
 * @param img_gpu_in
 * @param img_gpu_out
 * @param width
 * @param height
 */
__global__ void simple_transpose_kernel(float *img_gpu_in, float *img_gpu_out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        img_gpu_out[x * height + y] = img_gpu_in[x + y * width];
    }
}

/**
 * Improved transpose kernel
 * @param img_gpu_in
 * @param img_gpu_out
 * @param width
 * @param height
 */
__global__ void improved_transpose_kernel(float *img_gpu_in, float *img_gpu_out, int width, int height) {
    __shared__ float m_block[BLOCK_SIZE * BLOCK_SIZE];

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // Load in shared memory
    bool compute = x < width && y < height;
    if (compute) {
        m_block[threadIdx.x + BLOCK_SIZE * threadIdx.y] = img_gpu_in[x + y * width];
    }

    __syncthreads();

    // Load in registry
    float transposed_value;
    if (compute) {
        transposed_value = m_block[threadIdx.y + threadIdx.x * BLOCK_SIZE];
    }

    // Store in transposed tile
    int tx = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    int ty = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    if (compute) {
        img_gpu_out[tx + ty * height] = transposed_value;
    }

}

/**
 * Improved transpose kernel
 * @param img_gpu_in
 * @param img_gpu_out
 * @param width
 * @param height
 */
__global__ void improved_transpose_dummy_kernel(float *img_gpu_in, float *img_gpu_out, int width, int height) {
    __shared__ float m_block[(BLOCK_SIZE + 1) * BLOCK_SIZE];

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // Load in shared memory
    bool compute = x < width && y < height;
    if (compute) {
        m_block[threadIdx.x + threadIdx.y * (BLOCK_SIZE + 1)] = img_gpu_in[x + y * width];
    }

    __syncthreads();

    // Load in registry
    float transposed_value;
    if (compute) {
        transposed_value = m_block[threadIdx.y + threadIdx.x * (BLOCK_SIZE + 1)];
    }

    // Store in transposed tile
    int tx = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    int ty = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    if (compute) {
        img_gpu_out[tx + ty * height] = transposed_value;
    }

}

// Kernel callers
void gpu_simple_transpose(const dim3 &gridSize, const dim3 &blockSize, float *img_gpu_in, float *img_gpu_out, int width, int height) {
    simple_transpose_kernel<<<gridSize, blockSize>>>(img_gpu_in, img_gpu_out, width, height);
    CUDA_CHK(cudaGetLastError())
    CUDA_CHK(cudaDeviceSynchronize())
}

void gpu_improved_transpose(const dim3 &gridSize, const dim3 &blockSize, float *img_gpu_in, float *img_gpu_out, int width, int height) {
    improved_transpose_kernel<<<gridSize, blockSize>>>(img_gpu_in, img_gpu_out, width, height);
    CUDA_CHK(cudaGetLastError())
    CUDA_CHK(cudaDeviceSynchronize())
}

void gpu_improved_transpose_dummy(const dim3 &gridSize, const dim3 &blockSize, float *img_gpu_in, float *img_gpu_out, int width, int height) {
    improved_transpose_dummy_kernel<<<gridSize, blockSize>>>(img_gpu_in, img_gpu_out, width, height);
    CUDA_CHK(cudaGetLastError())
    CUDA_CHK(cudaDeviceSynchronize())
}

// Utils
void allocate_and_copy_gpu(float* &gpu_in, float* &gpu_out, float *cpu_in, float *cpu_out, int width, int height) {
    size_t size = width * height * sizeof(float);
    CUDA_CHK ( cudaMalloc((void**)& gpu_in, size) )
    CUDA_CHK ( cudaMalloc((void**)& gpu_out, size) )

    CUDA_CHK ( cudaMemcpy(gpu_in, cpu_in, size, cudaMemcpyHostToDevice) )
    CUDA_CHK ( cudaMemcpy(gpu_out, cpu_out, size, cudaMemcpyHostToDevice) )
}

void copy_and_free_gpu(float* &gpu_in, float* &gpu_out, float *cpu_out, int width, int height) {
    size_t size = width * height * sizeof(float);
    CUDA_CHK ( cudaMemcpy(cpu_out, gpu_out, size, cudaMemcpyDeviceToHost) )
    CUDA_CHK ( cudaFree(gpu_in) )
    CUDA_CHK ( cudaFree(gpu_out) )
}

double simple_transpose(float* in_cpu_m, float* out_cpu_m, int width, int height) {
    float * img_gpu = NULL, * img_gpu_out = NULL;
    allocate_and_copy_gpu(img_gpu, img_gpu_out, in_cpu_m, out_cpu_m, width, height);

    // TODO: Assume the image is multiple of BLOCK_SIZE
    dim3 gridSize( (int)((float)width)/BLOCK_SIZE, (int)((float)height)/BLOCK_SIZE ); // 40 x 30
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    MS(gpu_simple_transpose(gridSize, blockSize, img_gpu, img_gpu_out, width, height), time)

    copy_and_free_gpu(img_gpu, img_gpu_out, out_cpu_m, width, height);

    return time;
}

double improved_transpose(float* in_cpu_m, float* out_cpu_m, int width, int height) {
    float * img_gpu = NULL, * img_gpu_out = NULL;
    allocate_and_copy_gpu(img_gpu, img_gpu_out, in_cpu_m, out_cpu_m, width, height);

    // TODO: Assume the image is multiple of BLOCK_SIZE
    dim3 gridSize( (int)((float)width)/BLOCK_SIZE, (int)((float)height)/BLOCK_SIZE ); // 40 x 30
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    MS(gpu_improved_transpose(gridSize, blockSize, img_gpu, img_gpu_out, width, height), time)

    copy_and_free_gpu(img_gpu, img_gpu_out, out_cpu_m, width, height);

    return time;
}

double improved_transpose_dummy(float* in_cpu_m, float* out_cpu_m, int width, int height) {
    float * img_gpu = NULL, * img_gpu_out = NULL;
    allocate_and_copy_gpu(img_gpu, img_gpu_out, in_cpu_m, out_cpu_m, width, height);

    // TODO: Assume the image is multiple of BLOCK_SIZE
    dim3 gridSize( (int)((float)width)/BLOCK_SIZE, (int)((float)height)/BLOCK_SIZE ); // 40 x 30
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    MS(gpu_improved_transpose_dummy(gridSize, blockSize, img_gpu, img_gpu_out, width, height), time)

    copy_and_free_gpu(img_gpu, img_gpu_out, out_cpu_m, width, height);

    return time;
}

