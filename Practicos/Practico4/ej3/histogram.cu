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

    if (x < width && y < height) {
        int color = (int)img_gpu_in[x + y * width];
        atomicAdd(&img_gpu_out[color], 1.f);
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

    if (x < width && y < height) {
        int color = (int)img_gpu_in[x + y * width];
        atomicAdd(&h_block[color], 1.f);
    }

    __syncthreads();

    if (tid < COLOR_SIZE) {
        atomicAdd(&img_gpu_out[tid], h_block[tid]);
    }
}

__global__ void matrix_histogram_kernel(float *img_gpu_in, float *img_gpu_out, int width, int height) {
    __shared__ float h_block[COLOR_SIZE];

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int bid = blockIdx.x + blockIdx.y * gridDim.x;
    if (tid < COLOR_SIZE) {
        h_block[tid] = 0.f;
    }

    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int color = (int)img_gpu_in[x + y * width];
        atomicAdd(&h_block[color], 1.f);
    }

    __syncthreads();

    if (tid < COLOR_SIZE) {
        img_gpu_out[tid + bid * COLOR_SIZE] = h_block[tid];
    }
}

#define COLOR_PER_BLOCK 4
#define REDUCE_BLOCK_HEIGHT 128

__global__ void matrix_histogram_reduce_kernel(float *i_histogram_m, float *o_histogram_m, size_t height, size_t width, size_t size) {
    extern __shared__ float h_block[];

    // Initialize shared memory
    size_t tid_init = threadIdx.x + threadIdx.y * blockDim.x;
    if (tid_init < size) {
        h_block[tid_init] = 0.f;
        h_block[tid_init + blockDim.x * blockDim.y] = 0.f;
    }

    __syncthreads();

    // Copy to shared memory
    // width should be 256 color size
    size_t g_mem_a = blockIdx.y * blockDim.y * width * 2 + threadIdx.y * width + threadIdx.x + blockIdx.x * blockDim.x;
    size_t g_mem_b = g_mem_a + width * blockDim.y;

    size_t tid = threadIdx.x * blockDim.y * 2 + threadIdx.y; // tid on the shared block (transposed)

    if (g_mem_a < height * width) {
        h_block[tid] = i_histogram_m[g_mem_a];
    }
    if (g_mem_b < height * width) {
        h_block[tid + blockDim.y] = i_histogram_m[g_mem_b];
    }

    __syncthreads();

    // Reduce
    size_t reduce_width = fminf(blockDim.y * 2, height);
    int j = reduce_width;
    while (j > 1)
    {
        reduce_width = (j + 1) / 2; // integer ceil
        if (threadIdx.y < reduce_width && (threadIdx.y + reduce_width) < j) {
            h_block[tid] += h_block[tid + reduce_width];
        }
        j = reduce_width;
        __syncthreads();
    }

    __syncthreads();

    int color = threadIdx.x + blockIdx.x * blockDim.x;
    // Write to global memory
    if (threadIdx.y == 0) {
        o_histogram_m[blockIdx.y * 256 + color] = h_block[tid];
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
        default:
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

/**
 * This is for the kernel with a matrix of histogram (ex b)
 */

#define COLORS_PER_BLOCK 4
#define BLOCKS_PER_BLOCK 64  // Must be divisible by 2

bool execute_kernel_histogram(int width, int height, float * img_gpu, float * img_gpu_hist, float * img_gpu_hist_b) {
    // Grid
    dim3 gridSize( (int)((float)width)/BLOCK_SIZE, (int)((float)height)/BLOCK_SIZE );
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    size_t block_qty = (int)((float)width)/BLOCK_SIZE * (int)((float)height)/BLOCK_SIZE;
    // Execute
    // Create histogram matrix
    matrix_histogram_kernel<<<gridSize, blockSize>>>(img_gpu, img_gpu_hist, width, height);

    bool swapped = false;
    int reduceWidth = COLOR_SIZE;
    int reduceHeight = block_qty;

    dim3 reduceBlockSize(COLORS_PER_BLOCK, BLOCKS_PER_BLOCK / 2);
    size_t sharedMemSize = BLOCKS_PER_BLOCK * COLORS_PER_BLOCK * sizeof(int);

    do
    {
        dim3 reduceGridSize( (int)ceil(((float)reduceWidth)/COLORS_PER_BLOCK), (int)ceil(((float)reduceHeight)/BLOCKS_PER_BLOCK)); // Multiplied 2 because each block sums "another" block

        CUDA_CHK(cudaDeviceSynchronize())
        if (!swapped)
            matrix_histogram_reduce_kernel<<<reduceGridSize, reduceBlockSize, sharedMemSize>>>(img_gpu_hist, img_gpu_hist_b, reduceHeight, reduceWidth, block_qty * 256);
        else
            matrix_histogram_reduce_kernel<<<reduceGridSize, reduceBlockSize, sharedMemSize>>>(img_gpu_hist_b, img_gpu_hist, reduceHeight, reduceWidth, block_qty * 256);

        CUDA_CHK(cudaGetLastError())

        swapped = !swapped;

        reduceHeight = ceil((float)reduceHeight / BLOCKS_PER_BLOCK);

    } while (1 < reduceHeight);

    CUDA_CHK(cudaDeviceSynchronize())

    return swapped;
}

double execute_histogram(float* in_cpu_m, float* out_cpu_m, int width, int height) {
    // img_gpu is the same as in previous exercises, img_gpu_out should be a matrix of histograms 256 * block_qty
    float * img_gpu = NULL, * img_gpu_hist = NULL, *img_gpu_hist_b = NULL;

    size_t in_size = width * height * sizeof(float);
    size_t block_qty = (int)((float)width)/BLOCK_SIZE * (int)((float)height)/BLOCK_SIZE;
    size_t hist_size = block_qty * COLOR_SIZE * sizeof(float);

    // Allocate
    CUDA_CHK ( cudaMalloc((void**)& img_gpu, in_size) )
    CUDA_CHK ( cudaMalloc((void**)& img_gpu_hist, hist_size) )
    CUDA_CHK ( cudaMalloc((void**)& img_gpu_hist_b, hist_size) )
    CUDA_CHK ( cudaMemcpy(img_gpu, in_cpu_m, in_size, cudaMemcpyHostToDevice) )
    CUDA_CHK ( cudaMemset(img_gpu_hist, 0, hist_size) )  // Initialize gpu_out in 0

    bool swapped;
    MS(swapped = execute_kernel_histogram(width, height, img_gpu, img_gpu_hist, img_gpu_hist_b), time)

    if (!swapped)
        CUDA_CHK ( cudaMemcpy(out_cpu_m, img_gpu_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost) )
    else
        CUDA_CHK ( cudaMemcpy(out_cpu_m, img_gpu_hist_b, 256 * sizeof(int), cudaMemcpyDeviceToHost) )

    CUDA_CHK ( cudaFree(img_gpu) )
    CUDA_CHK ( cudaFree(img_gpu_hist) )
    CUDA_CHK ( cudaFree(img_gpu_hist_b) )

    return time;
}