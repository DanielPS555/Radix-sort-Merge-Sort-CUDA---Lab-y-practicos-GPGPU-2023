#include <stdio.h>

#include "include/scan.h"

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

__device__
void exlusiveScan(int valor, int * valorSalida){
    int offset = 1;
    int lane = threadIdx.x;

    // 1º Etapa
    while (offset < WARP_SIZE ) {
        int preValor = __shfl_up_sync(FULL_MASK, valor, offset);
        if ( ((lane + 1) % (offset*2)) == 0 )
            valor += preValor;
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

    valorSalida = valor;
}

void pruebaScan(int * srcCpu, int * dstCpu){
    int * srcGpu = NULL, *dstGpu = NULL;

    //allocate
    size_t size = 32 * sizeof (int);
    CUDA_CHK( cudaMalloc ((void **)& srcGpu , size ) )
    CUDA_CHK( cudaMalloc ((void **)& dstGpu , size ))

    CUDA_CHK( cudaMemcpy(srcGpu, srcCpu, size, cudaMemcpyHostToDevice))

    dim3 gridSize ( 1, 1);
    dim3 blockSize (32, 1);

    callPruebaScan<<<gridSize, blockSize>>>(srcGpu, dstGpu,0,0);
    CUDA_CHK(cudaGetLastError())
    CUDA_CHK(cudaDeviceSynchronize())

    CUDA_CHK( cudaMemcpy(dstCpu, dstGpu, size, cudaMemcpyDeviceToHost))
    CUDA_CHK ( cudaFree(srcGpu) )
    CUDA_CHK ( cudaFree(dstGpu) )

}
