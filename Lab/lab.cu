#include <stdio.h>

#include "include/lab.h"

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
 * En base a dos array ordenados CONSECUTIVOS de largo largoA y largoB respectivamente
 * escribe en "array" el nuevo array ordenado producto de ordenar los dos anteriores
 * @param array
 * @param largoA
 * @param largoB
 */
__global__
void orderedJoin(int * src, int largoA, int largoB){
    extern __shared__ int shared[]; //Size = largoA + largob . Se almacena de forma compartida la informacion de los dos arrays a juntar
    //Voy a destinar los primeros "largoA" threads para ordenar los elementos de A, los demas para ordenar los de B
    bool soyDeB = threadIdx.x >= largoA;
    int idEnArray = soyDeB ? threadIdx.x - largoA: threadIdx.x; //Es la posicion dentro de A o B respectivamente
    int valor = src[ blockIdx.x * blockDim.x + threadIdx.x ]; //Todo analizar viabilidad de hacer consulta coalleced
    shared[threadIdx.x] = valor;

    __syncthreads();

    // Nota: Aqui hay que tener un cuidado adicional, ¿Que pasa si en la misma posiicon i de A y B tenemos que r(a_i, B) = r(b_i, A)?
    //         La solucion que encontramos es hacer que ese j no sea igual discriminando entre el array A y el B, dando la presetncia a A, mas detalle a continuacion:
    //         Notar que si a_i < b_i, entonces r(a_i, B) < r(b_i,B) = i = r(a_i, A) < r(b_i, A) (analogo para b_i < a_i), entonces aqui no hay problema.
    //         Pero lo mismo no ocurre si a_i = b_i = h, ya que dependiendo de la politica de la busqueda, tanto el a_i como el b_i tendrian como r(a_i, B) = r(b_i, A) = posicion al inicio (o final) de la rafaga de h
    //         Es por eso que se propone que la politica de busqueda de r(a_i, B) y r(b_i, A) sea diferente
    //         De forma tal que los elementos de A se "inserten" previo a todos los valores iguales en B, mientras que para los de B buscaremos su posicion de forma tal que sea luego de los iguales en A
    //         Esto ultimo es lo que representa el ultimo parametro de el metodo busquedaPorBiparticion

    int posicionEnElOtroArray = busquedaPorBiparticion(shared, soyDeB ? 0 : largoA, soyDeB ? largoA : largoB, valor, !soyDeB);

    __syncthreads();

    shared[idEnArray + posicionEnElOtroArray] = valor; //Escribo mi valor en la nueva posicion

    __syncthreads();

    src[ blockIdx.x * blockDim.x + threadIdx.x ] = shared[threadIdx.x];
}




/**
 * Threads will read
 */
 /*
__global__
void read_separators(int* a_in, int* b_in, int* s_out, int a_size, int b_size, int s_size, int t_size, int separator_count)
{
    // TODO: are s_size and separator_count the same?

    int section_id = blockIdx.x;  // A - B section
    int section_offset = section_id * separator_count;
    int separatorId = threadIdx.x + threadIdx.y * blockDim.x;

    if (separatorId >= separator_count)
        return;

    int a_index = separatorId * t_size;
    int b_index = a_index;

    if (separatorId < s_size)
    {
        // last segment gets the last element
        if ((s_size - 1) == separatorId)
        {
            a_index = a_size - 1;
            b_index = b_size - 1;
        }

        if (a_index < a_size)
            s_out[section_offset + separatorId] = a_in[a_index];

        if (index < b_size)
            s_out[section_offset + separatorId + separator_count] = b_in[b_index];
    }
} */


/**
 * Read separators from an input buffer.
 * @param data_in Input buffer
 * @param data_size Size of the input buffer
 * @param separators_out Output buffer with each separator
 * @param separators_size Size of the output buffer separators_out for all separators in all segments
 * @param sector_size Size of the sector A + B
 * @param separators_per_sector Total amount of separators per sector
 * The kernel will read both A and B buffers and will write the separators in the output buffer.
 */
__global__
void read_separators(int * data_in, size_t data_size, int * separators_out, size_t separators_size, int sector_size, int separators_per_sector) {
    // Asumamos por ahora que no hay casos donde el tamaño de los datos no sea multiplo del tamaño de la seccion
    //int separatorId = threadIdx.x + threadIdx.y * blockDim.x + SECTION_ID * SECTION_SIZE;
#define THREAD_ID threadIdx.x + threadIdx.y * blockDim.x
#define BLOCK_ID  blockIdx.x + blockIdx.y * gridDim.x
#define BLOCK_DIM blockDim.x * blockDim.y

    int segment_size = sector_size / separators_per_sector;

    // unique id for the thread
    int uid = THREAD_ID + BLOCK_ID * BLOCK_DIM;
    // based on the unique id we find the sector_id. The id of the AB sector
    int sector_id = uid / sector_size;
    // separator in the sector
    int separator_id = uid % separators_per_sector;
    // offset for the data array
    int sector_offset_d = sector_id * sector_size;

    // int a_index = separatorId * t_size + SECTION_ID * segment_size * 2;
    int a_index = sector_offset_d + separator_id * segment_size;
    // int b_index = a_index + segment_size;
    int b_index = a_index + sector_size / 2;

    // offset para acceder al array de separadores
    int sector_offset_s = separators_per_sector * sector_id;

    if (separator_id < separators_per_sector / 2 && sector_offset_s + separator_id + separators_per_sector / 2 < separators_size) {
        if (a_index < data_size)
            separators_out[sector_offset_s + separator_id] = data_in[a_index];

        if (b_index < data_size)
            separators_out[sector_offset_s + separator_id + separators_per_sector / 2] = data_in[b_index];
    }
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

        orderedJoin<<<gridSizeOrdenerJoin, blockSizeOrdererJoin, blockSize*sizeof(int)>>>(srcGpu, blockSize/2, blockSize/2);
        CUDA_CHK(cudaGetLastError())
        CUDA_CHK(cudaDeviceSynchronize())

        blockSize *=2;
    }

    CUDA_CHK( cudaMemcpy(srcCpu, srcGpu, size, cudaMemcpyDeviceToHost))
    CUDA_CHK ( cudaFree(srcGpu) )
}

void test_secuence_reading (int * srcCpu, int length){

    int blockSize = 256; // se ejecuta test_with_block_under_256 antes

    int * srcGpu = NULL;

    size_t size = length * sizeof (int);
    CUDA_CHK( cudaMalloc ((void **)& srcGpu , size ) )

    CUDA_CHK( cudaMemcpy(srcGpu, srcCpu, size, cudaMemcpyHostToDevice))

    if (blockSize <= length) {

        int segment_count = length / (256 / 2); // How many segments of 256/2 are there

        // so the seprators are always going to be the same, only their vales are going to change
        int* gpu_segment_values;
        CUDA_CHK( cudaMalloc ((void **)& gpu_segment_values , segment_count * sizeof(int) ) )
        int* cpu_segment_values = (int*) malloc(segment_count * sizeof(int));

        // read separators
        // size of each A + B
        int sector_size = blockSize * 2;
        int t = blockSize / 2;


        //while (segment_size <= length) {


            // int section_qty = length / (blockSize * 2); // How many ab groups are there

            //dim3 gridSize(section_qty, 1);
            dim3 dimBlockSize(32, 32);
            dim3 gridSize((32 * 32 + segment_count - 1) / segment_count, 1);
            int separators_per_sector = sector_size / t;

            read_separators<<<gridSize, dimBlockSize>>>(srcGpu, length, gpu_segment_values, segment_count, sector_size, separators_per_sector);

            CUDA_CHK(cudaGetLastError())
            CUDA_CHK(cudaDeviceSynchronize())

            // foreach sector
            for (int i = 0; i < segment_count; i++) {
                printf("%d ", cpu_segment_values[i]);
            }


            CUDA_CHK( cudaMemcpy(cpu_segment_values, gpu_segment_values, segment_count * sizeof(int), cudaMemcpyDeviceToHost) )
            for (int i = 0; i < segment_count; i++) {
                printf("%d ", cpu_segment_values[i]);
            }



            // encuentro separadores
            // Sa + Sb
            // sector_size *= 2;

        //}



        printf("\n");

        free(cpu_segment_values);
        CUDA_CHK( cudaFree(gpu_segment_values) )

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
