#include <stdio.h>

#include "include/lab.h"
#include <thrust/sort.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
#define NONE_MASK 0x00000000
#define RADIX_SORT_BLOCK_SIZE 256

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




/*
__device__
int split(int lane, int currentValue, int mask){



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
*/

__global__
void radix_sort_kernel(int * src){

    __shared__ int swap [RADIX_SORT_BLOCK_SIZE];

    int lane = threadIdx.x % WARP_SIZE; //Se asume que los bloques son multiplos de 32, por lo que para obtener mi lane no presiso saber nada mas que mi modulo 32.
    int tie = threadIdx.x / WARP_SIZE;
    int idInArray = threadIdx.x + blockDim.x * blockIdx.x;
    int currentValue = src[idInArray];

    int valorThreadPrevio = __shfl_up_sync(FULL_MASK, currentValue, 1); //Obtengo el valor de mi lane previo
    if (lane == 0)
        valorThreadPrevio = currentValue - 1; //En caso que sea el primer lane, hago que mi "anterior" numero sea siempre menor, en particular uno menor

    int mask = 1;
    while (__all_sync(FULL_MASK,valorThreadPrevio <= currentValue) == 0){ //La operacion termina cuando el warp esta ordenado

        int notValidateMask = (currentValue & mask) == 0 ? 1 : 0;
        int valueInScan = exlusiveScan(notValidateMask);

        int totalFalse = notValidateMask + valueInScan; //Este valor es invalido para todos menos el ultimo lane, por eso en la siguiente linea todos se lo pide
        totalFalse = __shfl_sync(FULL_MASK, totalFalse, WARP_SIZE - 1); //Todos los lane obtiene el valor correcto del ultimo lane

        int t = lane - valueInScan + totalFalse;
        int nuevaPosicion = notValidateMask ? valueInScan : t;
        swap[tie + nuevaPosicion] = currentValue;

        __syncwarp();

        currentValue = swap[tie + lane];

        mask <<= 1; //Muevo la mascara

        valorThreadPrevio = __shfl_up_sync(FULL_MASK, currentValue, 1);
        if (lane == 0)
            valorThreadPrevio = currentValue - 1;
    }

    src[idInArray] = currentValue;
}


/**
 * Dado un array ordenado desde [posicionInicio, posicionInicio + size ], devuelve la posicion en la que deberia ser insertado objetoBuscado
 * Nota: si previoAIguales == true, este debuelbe la posicion de forma que objetoBuscado sea insertado antes que los iguales,
 *  en caso contrario sera la posicion de ser insertado detras de los iguales
 *
 *  Retorna la posicion relativa dentro del arreglo. O sea seria la posicion real dentro del array, menos el inicio
 * @param posicionInicio
 * @param size
 * @param objetoBuscado
 * @param previoAIguales
 * @return
 */

__device__
int busquedaPorBiparticion(int * array, int posicionInicio, int size, int objetoBuscado, bool previoAIguales){

    if(size <= 0) return 0;

    int final = posicionInicio + size - 1;
    int inicio = posicionInicio;

    int medio = (inicio + final) /2;

    while (inicio < final){ //Hay mas de un elemento en el area de busqueda

        int pivot = array[medio];

        bool buscarAbajo = previoAIguales ? objetoBuscado <= pivot : objetoBuscado < pivot;

        inicio = buscarAbajo ? inicio : medio + 1;
        final  = buscarAbajo ? medio  : final;
        medio  = (inicio + final) /2;

    }

    int pivot = array[medio];

    if (previoAIguales){
        return (pivot < objetoBuscado ? medio + 1 : medio) - posicionInicio;
    } else {
        return (pivot > objetoBuscado ? medio : medio + 1) - posicionInicio;
    }

}


/**
 * Se encarga de convinar el array
 * @param arrayA
 * @param largoA
 * @param arrayB
 * @param largoB
 * @param arraySalida
 * @param sharedToUse
 */
__device__
void deviceOrderedJoin(int * src, int posicionLecturaA, int largoA, int posicionLecturaB, int largoB, int * arraySalida, int posicionSalida, int * sharedToUse){

    //En una primera instancia voy a destinar los primeros "largoPorSegmento" threads para ordenar los elementos de A, los demas para ordenar los de B.
    // ToDo utilizar la idea del ej 2 practico 4 para hacer que la lectura sea coalesed

    bool soyDeB;
    int idMiArray;
    int valor;
    if(threadIdx.x < largoA + largoB){
        soyDeB = threadIdx.x >= largoA;
        idMiArray = soyDeB ? threadIdx.x - largoA: threadIdx.x; //Es la posicion dentro de A o B respectivamente (sin contar el offset de posicionLecturaA o posicionLecturaB)
        valor = soyDeB ? src[posicionLecturaB + idMiArray] : src[posicionLecturaA + idMiArray];
        sharedToUse[threadIdx.x] = valor;
    }
    __syncthreads();

    int posicionEnElOtroArray;
    if(threadIdx.x < largoA + largoB) {
        // Nota: Aqui hay que tener un cuidado adicional, ¿Que pasa si en la misma posiicon i de A y B tenemos que r(a_i, B) = r(b_i, A)?
        //         La solucion que encontramos es hacer que ese j no sea igual discriminando entre el array A y el B, dando la presetncia a A, mas detalle a continuacion:
        //         Notar que si a_i < b_i, entonces r(a_i, B) < r(b_i,B) = i = r(a_i, A) < r(b_i, A) (analogo para b_i < a_i), entonces aqui no hay problema.
        //         Pero lo mismo no ocurre si a_i = b_i = h, ya que dependiendo de la politica de la busqueda, tanto el a_i como el b_i tendrian como r(a_i, B) = r(b_i, A) = posicion al inicio (o final) de la rafaga de h
        //         Es por eso que se propone que la politica de busqueda de r(a_i, B) y r(b_i, A) sea diferente
        //         De forma tal que los elementos de A se "inserten" previo a todos los valores iguales en B, mientras que para los de B buscaremos su posicion de forma tal que sea luego de los iguales en A
        //         Esto ultimo es lo que representa el ultimo parametro de el metodo busquedaPorBiparticion
        posicionEnElOtroArray = busquedaPorBiparticion(sharedToUse, soyDeB ? 0 : largoA, soyDeB ? largoA : largoB, valor, !soyDeB);
    }

    __syncthreads();
    if(threadIdx.x < largoA + largoB) {
        sharedToUse[idMiArray + posicionEnElOtroArray] = valor; //Escribo mi valor en la nueva posicion
    }

    __syncthreads();
    if(threadIdx.x < largoA + largoB) {
        arraySalida[posicionSalida + threadIdx.x] = sharedToUse[threadIdx.x]; //Escritura en memoria global coaleced si posicionSalida es multiplo de 32
    }
}


/**
 * En base a dos array ordenados CONSECUTIVOS de largo largoA y largoB respectivamente
 * escribe en "array" el nuevo array ordenado producto de ordenar los dos anteriores
 * @param array
 * @param largoA
 * @param largoB
 */
__global__
void orderedJoin(int * src, int largoPorSegmento){
    extern __shared__ int shared[]; //Size = 2*largoPorSegmento . Se almacena de forma compartida la informacion de los dos arrays a juntar
    int posInicioBloque = blockIdx.x * blockDim.x;
    deviceOrderedJoin(src, posInicioBloque, largoPorSegmento, posInicioBloque + largoPorSegmento, largoPorSegmento, src, posInicioBloque, shared);
}

/**
 * Combina las secuencias delimitadas por posSeparadoresEnA y posSeparadoresEnB de A y B respectivamente,
 * las escribe en su respectiva posicion en el array dst
 *
 * Precondiciones:
 * - A y B, se encuentan consecutivos en src,
 * - length(A) = length(B) = largo
 * - A comienza en 2 * largo * blockIdx.y
 * - length(src) = length(dst)
 * - La grilla de bloques es dimencional, el dimencion y identifica al A y B a comvinar, mientras que la dimencion x identifica al par de separadores (segmento) que hay que juntar
 *
 * @param posSeparadoresEnA
 * @param posSeparadoresEnB
 * @param src
 * @param dst
 * @param largo
 */
__global__
void mergeSegmentUsingSeparators(int * posSeparadoresEnA, int * posSeparadoresEnB, int * src , int * dst, int largo){
    __shared__ int shared[512]; //Max que puede ocupar A + B
    int numeroDeSegmentosPorParte = gridDim.x; // La dimencion en X nos da el numero de segmentos

    int idParte = blockIdx.y;
    int idSegmento = blockIdx.x;

    int inicioA = posSeparadoresEnA[idParte * numeroDeSegmentosPorParte + idSegmento];
    int inicioB = posSeparadoresEnB[idParte * numeroDeSegmentosPorParte + idSegmento];

    int finA;
    int finB;
    if(idSegmento == numeroDeSegmentosPorParte - 1){ // Si soy el ultimo segmento de una parte
        finA = largo;
        finB = largo;
    } else {
        finA = posSeparadoresEnA[idParte * numeroDeSegmentosPorParte + idSegmento + 1];
        finB = posSeparadoresEnB[idParte * numeroDeSegmentosPorParte + idSegmento + 1];
    }

    int inicioDeLaParteEnSrc = idParte * 2 * largo;
    deviceOrderedJoin(src, inicioDeLaParteEnSrc + inicioA, max(0, finA - inicioA), inicioDeLaParteEnSrc + largo + inicioB, max(0, finB - inicioB), dst, inicioDeLaParteEnSrc + inicioA + inicioB,shared );
}

void test_with_block_under_256(int * srcCpu, int length){
    int * srcGpu = NULL;

    size_t size = length * sizeof (int);
    CUDA_CHK( cudaMalloc ((void **)& srcGpu , size ) )

    CUDA_CHK( cudaMemcpy(srcGpu, srcCpu, size, cudaMemcpyHostToDevice))

    //Etapa 1: Comienzo por el radixSort

    dim3 gridSizeRadixSort ( length / 32, 1);
    dim3 blockSizeRadixSort (32, 1);

    radix_sort_kernel<<<gridSizeRadixSort, blockSizeRadixSort>>>(srcGpu);

    CUDA_CHK(cudaGetLastError())
    CUDA_CHK(cudaDeviceSynchronize())

    //Etapa 2: Voy haciando merge sort hasta termino o llego a un bloque de 256

    int blockSize = 64; //32*2;
    while (blockSize <= min(256, length)){

        dim3 gridSizeOrdenerJoin ( length / blockSize, 1);
        dim3 blockSizeOrdererJoin (blockSize, 1);

        orderedJoin<<<gridSizeOrdenerJoin, blockSizeOrdererJoin, blockSize*sizeof(int)>>>(srcGpu, blockSize/2);
        CUDA_CHK(cudaGetLastError())
        CUDA_CHK(cudaDeviceSynchronize())

        blockSize *=2;
    }

    CUDA_CHK( cudaMemcpy(srcCpu, srcGpu, size, cudaMemcpyDeviceToHost))
    CUDA_CHK ( cudaFree(srcGpu) )

}


void test_radix_sort(int * srcCpu){
    int * srcGpu = NULL;

    //allocate
    size_t size = 64 * sizeof (int);
    CUDA_CHK( cudaMalloc ((void **)& srcGpu , size ) )

    CUDA_CHK( cudaMemcpy(srcGpu, srcCpu, size, cudaMemcpyHostToDevice))


    dim3 gridSize ( 2, 1);
    dim3 blockSize (32, 1);

    radix_sort_kernel<<<gridSize, blockSize>>>(srcGpu);
    CUDA_CHK(cudaGetLastError())
    CUDA_CHK(cudaDeviceSynchronize())

    CUDA_CHK( cudaMemcpy(srcCpu, srcGpu, size, cudaMemcpyDeviceToHost))
    CUDA_CHK ( cudaFree(srcGpu) )
}

void test_merge_segment_using_separators(int * array, int largo, int * sa, int * sb, int maximoSoporadoPorMergeSort){
    int * srcGpu = NULL;
    int * dstGpu = NULL;

    int * separadoresAGpu = NULL;
    int * separadoresBGpu = NULL;

    int largoSeparadores = largo / (maximoSoporadoPorMergeSort / 2) + 2;

    printf("El numero de separadores es = %d \n" , largoSeparadores);

    //allocate
    size_t size = 64 * sizeof (int);
    size_t sizeSeparadores = largoSeparadores * sizeof(int);

    CUDA_CHK( cudaMalloc ((void **)& srcGpu , size ) )
    CUDA_CHK( cudaMalloc ((void **)& dstGpu , size ) )

    CUDA_CHK( cudaMalloc ((void **)& separadoresAGpu , sizeSeparadores ) )
    CUDA_CHK( cudaMalloc ((void **)& separadoresBGpu , sizeSeparadores ) )

    CUDA_CHK( cudaMemcpy(srcGpu, array, size, cudaMemcpyHostToDevice))
    CUDA_CHK( cudaMemcpy(separadoresAGpu, sa, sizeSeparadores, cudaMemcpyHostToDevice))
    CUDA_CHK( cudaMemcpy(separadoresBGpu, sb, sizeSeparadores, cudaMemcpyHostToDevice))

    dim3 gridSize ( largoSeparadores, 1);
    dim3 blockSize (maximoSoporadoPorMergeSort, 1);

    mergeSegmentUsingSeparators<<<gridSize, blockSize>>>(separadoresAGpu, separadoresBGpu, srcGpu, dstGpu, largo/2);
    CUDA_CHK(cudaGetLastError())
    CUDA_CHK(cudaDeviceSynchronize())

    CUDA_CHK( cudaMemcpy(array, dstGpu, size, cudaMemcpyDeviceToHost))
    CUDA_CHK ( cudaFree(srcGpu) )
    CUDA_CHK ( cudaFree(dstGpu) )
    CUDA_CHK ( cudaFree(separadoresAGpu) )
    CUDA_CHK ( cudaFree(separadoresBGpu) )
}


void order_with_trust(int * src, int length){
    thrust::sort(src, src + length);
}
