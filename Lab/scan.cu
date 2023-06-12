#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "math.h"
#include <algorithm>

#include "include/scan.h"

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
#define NONE_MASK 0x00000000


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


__global__
void pruebaScan(int * src, int * dst,int posInicioSrc, int posInicioDst ){
    exlusiveScan(src, dst, posInicioSrc, posInicioDst);
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

    pruebaScan<<<gridSize, blockSize>>>(srcGpu, dstGpu,0,0);
    CUDA_CHK(cudaGetLastError())
    CUDA_CHK(cudaDeviceSynchronize())

    CUDA_CHK( cudaMemcpy(dstCpu, dstGpu, size, cudaMemcpyDeviceToHost))
    CUDA_CHK ( cudaFree(srcGpu) )
    CUDA_CHK ( cudaFree(dstGpu) )

}
