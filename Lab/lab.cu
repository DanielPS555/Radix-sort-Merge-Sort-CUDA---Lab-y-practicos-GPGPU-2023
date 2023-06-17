#include <stdio.h>

#include "include/lab.h"

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
#define NONE_MASK 0x00000000


__device__
int  exlusiveScan(int valor){
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
    return valor;
}

/**
 * Coloca a todos los valores que no cumplen con la maskara previo de los que si cumplen, manteniendo el orden relativo
 */
__device__
int split(int lane, int currentValue, int mask){

    int notValidateMask = (currentValue & mask) == 0 ? 1 : 0;
    int valueInScan = exlusiveScan(notValidateMask);

    int totalFalse = notValidateMask + valueInScan; //Este valor es invalido para todos menos el ultimo lane, por eso en la siguiente linea todos se lo pide
    totalFalse = __shfl_sync(FULL_MASK, totalFalse, WARP_SIZE - 1); //Todos los lane obtiene el valor correcto del ultimo lane

    int t = lane - valueInScan + totalFalse;
    int nuevaPosicion = notValidateMask ? valueInScan : t;

    //------------------

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
    int nextValue = __shfl_down_sync(FULL_MASK, value, 1);

    if (wid < WARP_SIZE - 1 && value > nextValue) {
        ordered = false;
    }
}




/**
 * Ordena los warps de tamaño 32 utilizando raid sort
 * Se asume que los bloques son unidimencionales
 * @param src
 * @param dst
 */
__global__
void radix_sort_kernel(int * src, int * dst){
    int lane = threadIdx.x % WARP_SIZE; //Se asume que los bloques son multiplos de 32, por lo que para obtener mi lane no presiso saber nada mas que mi modulo 32.
    int idInArray = threadIdx.x + blockDim.x * blockIdx.x;
    int currentValue = src[idInArray];

    int valorAnterior = __shfl_up_sync(FULL_MASK, currentValue, 1); //Obtengo el valor de mi lane previo
    if (lane == 0)
        valorAnterior --; //En caso que sea el primer lane, hago que mi "anterior" numero sea siempre menor, en particular uno menor

    int mask = 1;

    while (__all(valorAnterior <= currentValue) == 0){ //La operacion termina cuando el warp esta ordenado

        currentValue = split(lane, currentValue, mask);

        mask <<= 1; //Muevo la mascara

        int valorAnterior = __shfl_up_sync(FULL_MASK, currentValue, 1);
        if (lane == 0)
            valorAnterior --;
    }


    dst[idInArray] = currentValue;

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
