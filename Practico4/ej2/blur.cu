#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "math.h"
#include <algorithm>

#include "include/blur.h"

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


__global__ void blur_with_shared_memory(float * src, float * dst, int width, int heigth, int k){
    extern __shared__ float imagenShared[]; //Las dimenciones son ( 32 + 2k ) x ( 16 + 2k )

    int ventana_size_x = blockDim.x + 2*k;

    int length_dummy_alineacion = 0; //Como los bloques son de 32, el 1º pixel esta al inicio de un segmento en memoria gloval, al restar k no multiplo de 32, ya no se lee de memoria alineado
    if(k % 32 != 0)
        length_dummy_alineacion = 32 - k % 32;

    int search_area_posicion_y =  blockIdx.y * blockDim.y - k; //Posicion en la imagen, del inicio del vertical del area que leemos de global para pasar a cache
    int search_area_posicion_x =  blockIdx.x * blockDim.x - k - length_dummy_alineacion; //Posicion en la imagen, inicio horizontal del area que leemos de memoria global, no pasamos los primeros ni ultimos length_dummy_alineacion a compartida
    int search_area_size_y = blockDim.y + 2*k;
    int search_area_size_x = blockDim.x + 2*(k + length_dummy_alineacion);

    int idInterno = threadIdx.y * blockDim.x + threadIdx.x; //Este id identifica a uno de los thread dentro del bloque, por medio de este id, se les asigna pixeles a leer de global, de forma que se leea coaleced
    for (int i = idInterno; i < search_area_size_y * search_area_size_x; i += blockDim.x * blockDim.y){
        /*
         * Notemos que por la correcion por el length_dummy_alineacion, las filas de search_area es multiplo de 32, y esta alineado con respecto a memoria global
         *  por lo tanto, los thread de un mismo warp leen alineados
         * */
        int posx = search_area_posicion_x + i % search_area_size_x; //Ubicacion en la imagen en el eje x que esta leyendo el pixel

        if (posx < search_area_posicion_x + length_dummy_alineacion
             || posx > search_area_posicion_x + search_area_size_x - length_dummy_alineacion) //Estamos leyendo fuera de la ventana, en particular en el espacio dummy_alineacion
            continue;

        int posy = search_area_posicion_y + i / search_area_size_x;

        if (posx > width || posx < 0 || posy > heigth || posy < 0 ) //Recien controlamos que el pizel buscado este en la imagen aqui, al hacerlo aqui se simplifican las cuentas
            continue;

        float valorPixel = src[posy * width + posx];
        imagenShared[ (posy - search_area_posicion_y)*ventana_size_x + (posx - search_area_posicion_x - length_dummy_alineacion)] = valorPixel;
    }

    __syncthreads();

    int inicio_bloque_x = blockIdx.x * blockDim.x;
    int inicio_bloque_y = blockIdx.y * blockDim.y;
    int posicion_x = inicio_bloque_x + threadIdx.x;
    int posicion_y = inicio_bloque_y + threadIdx.y;

    int inicioVertical = max(0, posicion_y - k);
    int finVertical = min(heigth, posicion_y + k);
    int inicioHorizontal = max(0, posicion_x - k);
    int finHorizontal = min(width, posicion_x + k);

    int tam = (finVertical - inicioVertical + 1) * (finHorizontal - inicioHorizontal + 1);
    double acumulador = 0;
    for (int fila = inicioVertical; fila <= finVertical; fila++)
        for (int columna = inicioHorizontal; columna <= finHorizontal; columna++)
            acumulador += imagenShared[(fila + k - inicio_bloque_y)*ventana_size_x + (columna + k - inicio_bloque_x)]; //Recordar que la memoria compartida es del bloque,
                                                                                                                       // por lo que tengo que corregir por el inicio del bloque
                                                                                                                       // ademas como el bloque es de dimencion (32 + 2k) x ( 16 + 2k) tengo que sumar k a la posicion

    dst[ posicion_y * width + posicion_x] = acumulador/tam;

}


__global__ void blur_without_shared_memory(float * src, float * dst, int width, int height, int k){
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;

    int inicioVertical = max(0, pos_y - k);
    int finVertical = min(height, pos_y + k);
    int inicioHorizontal = max(0, pos_x - k);
    int finHorizontal = min(width, pos_x + k);

    int tam = (finVertical - inicioVertical + 1) * (finHorizontal - inicioHorizontal + 1);
    double acumulador = 0;
    for (int fila = inicioVertical; fila <= finVertical; fila++)
        for (int columna = inicioHorizontal; columna <= finHorizontal; columna++)
            acumulador += src[fila*width + columna];
    dst[pos_y*width + pos_x] = acumulador/tam;
}


void gpu_execute_kernel(algorithm_type algo, const dim3 &gridSize, const dim3 &blockSize, float *img_gpu_in, float *img_gpu_out, int width, int height, int k) {
    switch (algo) {
        case BLUR_WITHOUT_SHARED:
            blur_without_shared_memory<<<gridSize, blockSize>>>(img_gpu_in, img_gpu_out, width, height, k);
            break;
        case BLUR_WITH_SHARED:
            blur_with_shared_memory<<<gridSize, blockSize, (32+2*k)*(16+2*k)*sizeof(float)>>>(img_gpu_in, img_gpu_out, width, height, k); //ToDo buscar forma de sacar el hardcode del 32 y 16
            break;

    }
    CUDA_CHK(cudaGetLastError())
    CUDA_CHK(cudaDeviceSynchronize())
}

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


double execute_kernel(algorithm_type algo, float* in_cpu_m, float* out_cpu_m, int width, int height, int k){
    float * img_gpu = NULL, * img_gpu_out = NULL;
    allocate_and_copy_gpu(img_gpu, img_gpu_out, in_cpu_m, out_cpu_m, width, height);

    dim3 gridSize( (int)((float)width)/32, (int)((float)height)/16 ); //ToDo sacar hardcode
    dim3 blockSize(32, 16);

    MS(gpu_execute_kernel(algo, gridSize, blockSize, img_gpu, img_gpu_out, width, height, k), time)

    copy_and_free_gpu(img_gpu, img_gpu_out, out_cpu_m, width, height);

    return time;
}


