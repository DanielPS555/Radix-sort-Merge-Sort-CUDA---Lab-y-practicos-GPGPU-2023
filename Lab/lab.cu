#include <stdio.h>

#include "include/lab.h"

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
#define NONE_MASK 0x00000000

__device__
void exlusiveScan(int * src, int *dst, int posInicioSrc, int posInicioDst){
    int offset = 1;
    int lane = threadIdx.x;
    int valor = src[posInicioSrc + lane];


    // 1º Etapa

    while (offset < WARP_SIZE ) {
        int antValue = valor;
        valor += __shfl_up_sync(FULL_MASK, valor, offset);
        if ( ((lane + 1) % (offset*2)) != 0 ) { //ToDo preguntar al profe cual podria ser la estategia para mejorar esto
            valor = antValue;
        }
        offset *= 2;
    }

    if (lane == WARP_SIZE - 1){
        valor = 0;
    }

    // 2º Etapa
    offset /= 2;
    while (offset > 0 ) {
        int valorOffsetSuperior = __shfl_down_sync(FULL_MASK, valor, offset);
        int valorOffsetInferior = __shfl_up_sync(FULL_MASK, valor, offset);

        bool moduloDobleOffset = ((lane + 1) % (offset*2)) == 0;
        valor += moduloDobleOffset ? valorOffsetInferior : 0;
        valor = !moduloDobleOffset && ((lane + 1) % (offset)) == 0 ? valorOffsetSuperior : valor;

        offset /= 2;
    }

    dst[posInicioDst + lane] = valor;
}


/**
 * Order the array based on the byte mask.
 * @param array The array to be ordered (in shared memory). Use pointer arithmetic to privatize the array.
 * @param temp_array The temporary array to be used (in shared memory). Use pointer arithmetic to privatize the array.
 */
__device__
void split(int* array, int* prefix_array, int mask, bool &ordered)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x; // tid in block
    int wid = tid % WARP_SIZE; // id in warp

    int value = array[wid]; // value to be ordered (stored in order to reuse the "array")
    prefix_array[wid] = !(bool)(array[wid] & mask); // true if the bit is 0, false if the bit is 1

    // prefix sum
    exlusiveScan(prefix_array, array, 0, 0);

    int total_false = array[WARP_SIZE - 1] + prefix_array[WARP_SIZE - 1]; // total number of false
    int order_index = prefix_array[wid] ? array[wid] : total_false + wid - array[wid]; // order index

    // write the value in the correct position
    array[order_index] = value;

    // check if the array is ordered
    value = array[wid];
    int nextValue = __shfl_down_sync(0xffffffff, value, 1);

    if (wid < WARP_SIZE - 1 && value > nextValue) {
        ordered = false;
    }
}

#define BLOCK_SIZE 32

__global__
void test_radix_kernel(int * src, int * dst, size_t size){
    __shared__ int data   [BLOCK_SIZE * BLOCK_SIZE];  // shared memory to store the data to be sorted
    __shared__ int prefix [BLOCK_SIZE * BLOCK_SIZE];  // shared memory for temporal values
    __shared__ bool ordered [BLOCK_SIZE * BLOCK_SIZE / WARP_SIZE]; // shared memory to check if the array is ordered
    // load data
    int tid = threadIdx.x + threadIdx.y * blockDim.x; // tid in block

    if (tid < size && tid < BLOCK_SIZE * BLOCK_SIZE) {
        data[tid] = src[tid];
    }

    // initialize ordered array
    if (tid % WARP_SIZE == 0 && tid < size) {
        ordered[tid] = false;
    }

    int warp_id = tid / WARP_SIZE; // warp id

    __syncthreads();

    // TODO: Do it in while and test if it is ordered in each iteration with a shuffle
    int mask = 1;
    while (!ordered[warp_id])
    {
        // set ordered to true (it will be set to false if the array is not ordered)
        ordered[warp_id] = true;  // Shared memory, ignore any conflicts (only 1 write)
        split(data, prefix, mask, ordered[warp_id]);
        __syncthreads();  // this might not be needed
        mask <<= 1;
    }

    // write data
    if (tid < size) {
        dst[tid] = data[tid];  // TODO: we are assuming that the kernel runs with a single block
    }
}

void test_radix(int * srcCpu, int * dstCpu){
    int * srcGpu = NULL, *dstGpu = NULL;

    //allocate
    size_t size = 32 * sizeof (int);
    CUDA_CHK( cudaMalloc ((void **)& srcGpu , size ) )
    CUDA_CHK( cudaMalloc ((void **)& dstGpu , size ))

    CUDA_CHK( cudaMemcpy(srcGpu, srcCpu, size, cudaMemcpyHostToDevice))

    dim3 gridSize ( 1, 1);
    dim3 blockSize (32, 1);

    test_radix_kernel<<<gridSize, blockSize>>>(srcGpu, dstGpu, 32);
    CUDA_CHK(cudaGetLastError())
    CUDA_CHK(cudaDeviceSynchronize())

    CUDA_CHK( cudaMemcpy(dstCpu, dstGpu, size, cudaMemcpyDeviceToHost))
    CUDA_CHK ( cudaFree(srcGpu) )
    CUDA_CHK ( cudaFree(dstGpu) )
}
