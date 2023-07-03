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

/**
 * @param src Arreglo a ordenar
 * @param positions Posiciones del valor inicial, si el arreglo es NULL se hace el radix sort comun
 */
__global__
void radix_sort_kernel(int * src) {
    __shared__ int swap[RADIX_SORT_BLOCK_SIZE];

    int lane = threadIdx.x % WARP_SIZE; //Se asume que los bloques son multiplos de 32, por lo que para obtener mi lane no presiso saber nada mas que mi modulo 32.
    int tie = threadIdx.x / WARP_SIZE;
    int idInArray = threadIdx.x + blockDim.x * blockIdx.x;
    int currentValue = src[idInArray];

    int valorThreadPrevio = __shfl_up_sync(FULL_MASK, currentValue, 1); // Obtengo el valor de mi lane previo
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

        mask <<= 1; // Muevo la mascara

        valorThreadPrevio = __shfl_up_sync(FULL_MASK, currentValue, 1);

        if (lane == 0)
            valorThreadPrevio = currentValue - 1;
    }

    src[idInArray] = currentValue;
}


/**
 * Dado un array ordenado desde [posicionInicio, posicionInicio + size ], devuelve la posicion en la que deberia ser insertado objetoBuscado
 * Nota: si previoAIguales == true, este devuelve la posicion de forma que objetoBuscado sea insertado antes que los iguales,
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


#define BLOCK_ID  blockIdx.x  + blockIdx.y * gridDim.x
#define BLOCK_DIM blockDim.x  * blockDim.y

/**
 * One thread per separator.
 * Read each separator
 * Each block reads a sector
 * Returns 2 arrays with the A and B position of each separator
 */
__global__
void separators_kernel(int * in_data, int * out_separators_a, int * out_separators_b, const int sector_size, const int separators_per_sector, int t, int in_data_size) {
    extern __shared__ int separators[];
    int * separators_global_pos = separators + separators_per_sector;
    // based on the unique id we find the sector_id. The id of the AB sector
    const int sector_id = BLOCK_ID * blockDim.y + threadIdx.y;
    // separator in the sector
    const int separator_id = threadIdx.x;
    const bool is_a = separator_id < (separators_per_sector / 2);

    int segment_limit = min(sector_id * sector_size + (is_a ? (sector_size / 2) : sector_size), in_data_size);

    int pos = sector_id * sector_size + separator_id % (separators_per_sector / 2) * t + (is_a ? 0 : (sector_size / 2));
    // Last separator gets the last element
    if (pos >= segment_limit)
        pos = segment_limit - 1;

    int value = -1;
    if (separator_id < separators_per_sector)
    {
        value = in_data[pos];
        separators[separator_id] = value;
        separators_global_pos[separator_id] = pos;
    }

    __syncthreads();

    if (separator_id >= separators_per_sector)
        return;

    // Position in the other half of the separators array
    int s_offset = is_a ? (separators_per_sector / 2) : 0;
    int s_opp_position = busquedaPorBiparticion(separators, s_offset, separators_per_sector / 2, value, is_a);

    // if s_opp_position >= separators_per_sector / 2 then the end position is sector_id * sector_size + sector_size - 1
    // if s_opp_position <= 0 then the start position starts at the beginning of the opposite segment (A or B)
    int search_start, search_end;

    if (s_opp_position <= 0) {
        search_start = sector_id * sector_size + (is_a ? sector_size / 2 : 0);
    } else {
        search_start = separators_global_pos[s_opp_position + s_offset - 1];
    }

    if (s_opp_position >= (separators_per_sector / 2) - 1) {
        search_end = min(sector_id * sector_size + (is_a ? sector_size : sector_size / 2  ) - 1, in_data_size - 1);
    } else {
        search_end = separators_global_pos[s_opp_position + s_offset + 1];
    }

    // Find opposite position in in_data
    int opp_position = busquedaPorBiparticion(in_data, search_start, search_end - search_start, value, is_a) + search_start;

    const int pos_a = is_a ? pos : opp_position;
    const int pos_b = is_a ? opp_position : pos;

    // Ordered position in separators array out_separators_X
    int s_position = is_a ? (separator_id + s_opp_position) : (separator_id % (separators_per_sector / 2) + s_opp_position);

    out_separators_a[sector_id * separators_per_sector + s_position] = pos_a;
    out_separators_b[sector_id * separators_per_sector + s_position] = pos_b;
}

__global__
void merge_segments_kernel(int * separators_a, int * separators_b, int * src, int * dst, int sector_size) {
    __shared__ int shared[512];  // t * 2
    // Amount of segments per sector
    int segment_count = gridDim.x;
    int sector_id = blockIdx.y;
    int segment_id = blockIdx.x;  // 0-1, 1-2, 2-3, etc...

    int separator_position = sector_id * segment_count + segment_id;

    int start_a = separators_a[separator_position];
    int start_b = separators_b[separator_position];

    int sector_offset = sector_id * sector_size;

    // Repeat last element for last segment
    int end_a, end_b;
    if (segment_id == segment_count - 1) {
        end_a = sector_offset + sector_size / 2;
        end_b = sector_offset + sector_size;
    } else {
        end_a = separators_a[separator_position + 1];
        end_b = separators_b[separator_position + 1];
    }

    int dst_pos = start_a + start_b - sector_offset - sector_size / 2;

    deviceOrderedJoin(src, start_a, max(0, end_a - start_a), start_b, max(0, end_b - start_b), dst, dst_pos, shared);
}

#define MINIMUM_BLOCK_SIZE 32 // This is the minimum block size for all separator operations
void order_array(int * src_cpu, int length) {
    // 0 - Initialize GPU memory
    int * src_gpu = NULL;
    int * dst_gpu = NULL;
    int * separators_a_gpu = NULL;
    int * separators_b_gpu = NULL;

    size_t size = length * sizeof (int);
    size_t separators_size = length / MINIMUM_BLOCK_SIZE * sizeof (int);

    CUDA_CHK( cudaMalloc ((void **)& src_gpu , size ) )
    CUDA_CHK( cudaMalloc ((void **)& dst_gpu , size ) )
    CUDA_CHK( cudaMalloc ((void **)& separators_a_gpu , separators_size ) )
    CUDA_CHK( cudaMalloc ((void **)& separators_b_gpu , separators_size ) )

    CUDA_CHK ( cudaMemset(separators_a_gpu, 0, separators_size) )
    CUDA_CHK ( cudaMemset(separators_b_gpu, 0, separators_size) )

    CUDA_CHK( cudaMemcpy(src_gpu, src_cpu, size, cudaMemcpyHostToDevice))

    // 1 - Radix sort

    dim3 gridSizeRadixSort ( length / 32, 1);
    dim3 blockSizeRadixSort (32, 1);

    radix_sort_kernel<<<gridSizeRadixSort, blockSizeRadixSort>>>(src_gpu);
    CUDA_CHK(cudaGetLastError())
    CUDA_CHK(cudaDeviceSynchronize())

    // 2 - Merge sort on sectors smaller than a block

    int blockSize = 64; // 32 * 2;
    while (blockSize <= min(512, length)){
        dim3 gridSizeOrderedJoin ( length / blockSize, 1);
        dim3 blockSizeOrderedJoin (blockSize, 1);

        orderedJoin<<<gridSizeOrderedJoin, blockSizeOrderedJoin, blockSize*sizeof(int)>>>(src_gpu, blockSize/2);
        CUDA_CHK(cudaGetLastError())
        CUDA_CHK(cudaDeviceSynchronize())

        blockSize *=2;
    }

    blockSize /= 2;

    // 3 - Merge sort on sectors bigger than a block

    if (blockSize < length) {

        // Each A and B starts with the size of the block
        int sector_size = blockSize;  // Should start at 1024
        int t = blockSize / 2;

        int swapped = false;

        while (sector_size < length) {

            // 3.1 - Find separators

            sector_size *= 2;

            int sector_qty = (length + sector_size - 1) / sector_size;
            int separators_per_sector = 2*(1 + ((sector_size / 2) + t - 1)/t);

            int separators_count = sector_qty * separators_per_sector;

            int sectors_per_block = (separators_per_sector + 31) / 32;
            dim3 blockSizeFindSeparators (separators_per_sector, sectors_per_block);
            dim3 gridSizeFindSeparators ( (sector_qty + sectors_per_block - 1) / sectors_per_block, 1);

            size_t shared_size = separators_per_sector * sector_qty * sizeof(int) * 2;

            separators_kernel<<<gridSizeFindSeparators, blockSizeFindSeparators, shared_size>>>(src_gpu, separators_a_gpu, separators_b_gpu, sector_size, separators_per_sector, t, length);
            CUDA_CHK(cudaGetLastError())
            CUDA_CHK(cudaDeviceSynchronize())

            // 3.2 - Merge sort between separators

            dim3 mergeSegmentGridSize(separators_per_sector, sector_qty);
            dim3 mergeSegmentBlockSize(t * 2, 1);

            merge_segments_kernel<<<mergeSegmentGridSize, mergeSegmentBlockSize>>>(separators_a_gpu, separators_b_gpu, src_gpu, dst_gpu, sector_size);
            CUDA_CHK(cudaGetLastError())
            CUDA_CHK(cudaDeviceSynchronize())

            int * aux = src_gpu;
            src_gpu = dst_gpu;
            dst_gpu = aux;
            swapped = !swapped;
        }

    }

    // 4 - Copy result to CPU

    CUDA_CHK( cudaMemcpy(src_cpu, src_gpu, size, cudaMemcpyDeviceToHost))
    CUDA_CHK ( cudaFree(src_gpu) )
    CUDA_CHK ( cudaFree(dst_gpu) )
    CUDA_CHK ( cudaFree(separators_a_gpu) )
    CUDA_CHK ( cudaFree(separators_b_gpu) )
}

void order_with_trust(int * src, int length){
    thrust::sort(src, src + length);
}
