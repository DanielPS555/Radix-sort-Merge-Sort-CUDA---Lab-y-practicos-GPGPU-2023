
#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "math.h"
#include <algorithm>
using namespace std;


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

#define PRINT_CSV(name, width, height, time) printf("%s,%dx%d,%.4f\n", name, width, height, time);

// __global__ void blur_kernel(float* d_input, float* d_output, float* d_msk, int width, int height){

// }

__global__ void ajustar_brillo_coalesced_kernel(float* o_img, float* d_img, int width, int height, float coef){
    
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;

    int pos = pos_x + pos_y * width;

    if (pos_x < width && pos_y < height)
        d_img[pos] = min(255.0f,max(0.0f,o_img[pos]+coef));

}

__global__ void ajustar_brillo_no_coalesced_kernel(float* o_img, float* d_img, int width, int height, float coef){
    // Trasponemos las ubicaciones de forma tal que un mismo wap leean 32 filas diferentes
    int pos_y = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_x = blockIdx.y * blockDim.y + threadIdx.y;

    int pos = pos_x + pos_y * width;

    if (pos_x < width && pos_y < height)
        d_img[pos] = min(255.0f,max(0.0f,o_img[pos]+coef));

}

// Ej1b - 1
__global__ void ej1b_no_div_kernel(float* o_img, float* d_img, int width, int height, float coef){
    
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;

    int pos = pos_x + pos_y * width;

    int par = pos_x & 1;

    // Asumo tamaño de imagen multiplo de 64
    // pos_x = 2 * pos_x - 63 * par; // (pos_x - 32 * par) * 2 + par -> Equivale al if y else de la version div

    float value = sin(o_img[pos]) * par + cos(o_img[pos]) * (1 - par);
    d_img[pos] = min(255.0f,max(0.0f,value * 255.f));

}

// Ej1b - 2
__global__ void ej1b_div_kernel(float* o_img, float* d_img, int width, int height, float coef){
    
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;

    int pos = pos_x + pos_y * width;

    const float PI = 3.14159265358979323846f; // Ya llegamos a un nivel de desesperacion increible
    float cond = cos((pos_x & 1) * PI);

    if (pos_x < width && pos_y < height) {
        if (cond > 0.f) {
            float value = sin(o_img[pos]);
            d_img[pos] = min(255.0f, max(0.0f, value * 255.f));
        } else {
            float value = cos(o_img[pos]);
            d_img[pos] = min(255.0f, max(0.0f, value * 255.f));
        }
    }
}

// Ej2 - 1
__global__ void blur_gpu(float * img_in, int width, int height, float * img_out, int m_size){
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;

    int inicioVertical = max(0, pos_y - m_size);
    int finVertical = min(height, pos_y + m_size);
    int inicioHorizontal = max(0, pos_x - m_size);
    int finHorizontal = min(width, pos_x + m_size);
    
    int tam = (finVertical - inicioVertical + 1) * (finHorizontal - inicioHorizontal + 1);
    double acumulador = 0;
    for (int fila = inicioVertical; fila <= finVertical; fila++)
        for (int columna = inicioHorizontal; columna <= finHorizontal; columna++)
            acumulador += img_in[fila*width + columna];
    img_out[pos_y*width + pos_x] = acumulador/tam;
}

// Ej2 - 2
void blur_cpu(float * img_in, int width, int height, float * img_out, int m_size){
    for (int pos_x= 0; pos_x < width; pos_x++){
        for (int pos_y= 0; pos_y < height; pos_y++){
            int inicioVertical = max(0, pos_y - m_size);
            int finVertical = min(height - 1, pos_y + m_size);
            int inicioHorizontal = max(0, pos_x - m_size);
            int finHorizontal = min(width - 1, pos_x + m_size);
            
            int tam = (finVertical - inicioVertical + 1) * (finHorizontal - inicioHorizontal + 1);
            double acumulador = 0;

            for (int fila = inicioVertical; fila <= finVertical; fila++)
                for (int columna = inicioHorizontal; columna <= finHorizontal; columna++)
                    acumulador += img_in[fila*width + columna];

            img_out[pos_y*width + pos_x] = acumulador/tam;
        }    
    }

   
}

void ajustar_brillo_cpu(float * img_in, int width, int height, float * img_out, float coef){

    for(int imgx=0; imgx < width ; imgx++){
        for(int imgy=0; imgy < height; imgy++){
            img_out[imgy*width+imgx] = std::min(255.0f,std::max(0.0f,img_in[imgy*width+imgx]+coef));
        }
    }
    
}


void ejecutar_kernel_ajuste_brillo_coalesced_y_tomar_tiempo(dim3 gridSize, dim3 blockSize, float *img_gpu, float *img_gpu_out, int width, int heigth, float coeficiente){
    ajustar_brillo_coalesced_kernel<<<gridSize,blockSize >>>(img_gpu, img_gpu_out, width, heigth, coeficiente );
    CUDA_CHK( cudaGetLastError() );

    CUDA_CHK(cudaDeviceSynchronize())


}

void ejecutar_kernel_ajuste_brillo_no_coalesced_y_tomar_tiempo(dim3 gridSize, dim3 blockSize, float *img_gpu, float *img_gpu_out, int width, int heigth, float coeficiente){
    ajustar_brillo_no_coalesced_kernel<<<gridSize,blockSize >>>(img_gpu, img_gpu_out, width, heigth, coeficiente );
    CUDA_CHK( cudaGetLastError() );

    CUDA_CHK(cudaDeviceSynchronize())


}

void ejecutar_ej1b_no_div(dim3 gridSize, dim3 blockSize, float *img_gpu, float *img_gpu_out, int width, int heigth, float coeficiente){
    ej1b_no_div_kernel<<<gridSize,blockSize >>>(img_gpu, img_gpu_out, width, heigth, coeficiente );
    CUDA_CHK( cudaGetLastError() );

    CUDA_CHK(cudaDeviceSynchronize())

}

void ejecutar_ej1b_div(dim3 gridSize, dim3 blockSize, float *img_gpu, float *img_gpu_out, int width, int heigth, float coeficiente){
    ej1b_div_kernel<<<gridSize,blockSize >>>(img_gpu, img_gpu_out, width, heigth, coeficiente );
    CUDA_CHK( cudaGetLastError() );

    CUDA_CHK(cudaDeviceSynchronize())

}

void ejecutar_ej2_gpu(dim3 gridSize, dim3 blockSize, float *img_gpu, float *img_gpu_out, int width, int heigth, int k){
    blur_gpu<<<gridSize,blockSize >>>(img_gpu, width, heigth, img_gpu_out, k);
    CUDA_CHK( cudaGetLastError() );

    CUDA_CHK(cudaDeviceSynchronize())

}


void main_ajuste_brillo_cpu(float * img_in, int width, int height, float * img_out, float coef){
    MS(ajustar_brillo_cpu(img_in, width, height, img_out, coef), time);

    PRINT_CSV("ajuste_brillo_cpu", width, height, time);
}


void main_ajuste_brillo_coalesced(float *img_cpu, float *img_cpu_out, int width, int heigth, float coeficiente){

    float * img_gpu = NULL;
    float * img_gpu_out = NULL;

    size_t size = width * heigth * sizeof(float);
    CUDA_CHK ( cudaMalloc((void**)& img_gpu, size) );
    CUDA_CHK ( cudaMalloc((void**)& img_gpu_out, size) );

    CUDA_CHK ( cudaMemcpy(img_gpu, img_cpu, size, cudaMemcpyHostToDevice) );
    CUDA_CHK ( cudaMemcpy(img_gpu_out, img_cpu_out, size, cudaMemcpyHostToDevice) );


    dim3 gridSize( (int)((float)width)/32, (int)((float)heigth)/16 ); //ToDo Aca hay que ajustar bien los numeros
    dim3 blockSize(32, 16);

	MS(ejecutar_kernel_ajuste_brillo_coalesced_y_tomar_tiempo(gridSize, blockSize, img_gpu, img_gpu_out, width, heigth, coeficiente), time);

    PRINT_CSV("ajuste_brillo_coalesced", width, heigth, time)

    CUDA_CHK(cudaMemcpy(img_cpu_out, img_gpu_out, size, cudaMemcpyDeviceToHost));

    CUDA_CHK(cudaFree(img_gpu));
    CUDA_CHK(cudaFree(img_gpu_out));

}

void main_ajuste_brillo_no_coalesced(float *img_cpu, float *img_cpu_out, int width, int heigth, float coeficiente){

    float * img_gpu = NULL;
    float * img_gpu_out = NULL;

    size_t size = width * heigth * sizeof(float);
    CUDA_CHK ( cudaMalloc((void**)& img_gpu, size) );
    CUDA_CHK ( cudaMalloc((void**)& img_gpu_out, size) );

    CUDA_CHK ( cudaMemcpy(img_gpu, img_cpu, size, cudaMemcpyHostToDevice) );
    CUDA_CHK ( cudaMemcpy(img_gpu_out, img_cpu_out, size, cudaMemcpyHostToDevice) );

    dim3 gridSize((int)((float)heigth)/16 , (int)((float)width)/32 ); 
    dim3 blockSize(16, 32);

	MS(ejecutar_kernel_ajuste_brillo_no_coalesced_y_tomar_tiempo(gridSize, blockSize, img_gpu, img_gpu_out, width, heigth, coeficiente), time);

    PRINT_CSV("ajuste_brillo_no_coalesced", width, heigth, time)

    CUDA_CHK(cudaMemcpy(img_cpu_out, img_gpu_out, size, cudaMemcpyDeviceToHost));

    CUDA_CHK(cudaFree(img_gpu));
    CUDA_CHK(cudaFree(img_gpu_out));

}

void main_efecto_par_impar_divergente(float *img_cpu, float *img_cpu_out, int width, int heigth, float coeficiente){

    float * img_gpu = NULL;
    float * img_gpu_out = NULL;

    size_t size = width * heigth * sizeof(float);
    CUDA_CHK ( cudaMalloc((void**)& img_gpu, size) );
    CUDA_CHK ( cudaMalloc((void**)& img_gpu_out, size) );

    CUDA_CHK ( cudaMemcpy(img_gpu, img_cpu, size, cudaMemcpyHostToDevice) );
    CUDA_CHK ( cudaMemcpy(img_gpu_out, img_cpu_out, size, cudaMemcpyHostToDevice) );

    dim3 gridSize( (int)((float)width)/32, (int)((float)heigth)/16 ); //ToDo Aca hay que ajustar bien los numeros
    dim3 blockSize(32, 16);

	MS(ejecutar_ej1b_div(gridSize, blockSize, img_gpu, img_gpu_out, width, heigth, coeficiente), time);

    PRINT_CSV("efecto_par_impar_divergente", width, heigth, time)

    CUDA_CHK(cudaMemcpy(img_cpu_out, img_gpu_out, size, cudaMemcpyDeviceToHost));

    CUDA_CHK(cudaFree(img_gpu));
    CUDA_CHK(cudaFree(img_gpu_out));

}

void main_efecto_par_impar_no_divergente(float *img_cpu, float *img_cpu_out, int width, int heigth, float coeficiente){

    float * img_gpu = NULL;
    float * img_gpu_out = NULL;

    size_t size = width * heigth * sizeof(float);
    CUDA_CHK ( cudaMalloc((void**)& img_gpu, size) );
    CUDA_CHK ( cudaMalloc((void**)& img_gpu_out, size) );

    CUDA_CHK ( cudaMemcpy(img_gpu, img_cpu, size, cudaMemcpyHostToDevice) );
    CUDA_CHK ( cudaMemcpy(img_gpu_out, img_cpu_out, size, cudaMemcpyHostToDevice) );

    dim3 gridSize( (int)((float)width)/32, (int)((float)heigth)/16 ); //ToDo Aca hay que ajustar bien los numeros
    dim3 blockSize(32, 16);

	MS(ejecutar_ej1b_no_div(gridSize, blockSize, img_gpu, img_gpu_out, width, heigth, coeficiente), time);

    PRINT_CSV("efecto_par_impar_no_divergente", width, heigth, time)

    CUDA_CHK(cudaMemcpy(img_cpu_out, img_gpu_out, size, cudaMemcpyDeviceToHost));

    CUDA_CHK(cudaFree(img_gpu));
    CUDA_CHK(cudaFree(img_gpu_out));

}

void main_blur_gpu(float *img_cpu, float *img_cpu_out, int width, int heigth, int k){

    float * img_gpu = NULL;
    float * img_gpu_out = NULL;

    size_t size = width * heigth * sizeof(float);
    CUDA_CHK ( cudaMalloc((void**)& img_gpu, size) );
    CUDA_CHK ( cudaMalloc((void**)& img_gpu_out, size) );

    CUDA_CHK ( cudaMemcpy(img_gpu, img_cpu, size, cudaMemcpyHostToDevice) );
    CUDA_CHK ( cudaMemcpy(img_gpu_out, img_cpu_out, size, cudaMemcpyHostToDevice) );

    dim3 gridSize( (int)((float)width)/32, (int)((float)heigth)/16 ); //ToDo Aca hay que ajustar bien los numeros
    dim3 blockSize(32, 16);

	ejecutar_ej2_gpu(gridSize, blockSize, img_gpu, img_gpu_out, width, heigth, k);

    CUDA_CHK(cudaMemcpy(img_cpu_out, img_gpu_out, size, cudaMemcpyDeviceToHost));

    CUDA_CHK(cudaFree(img_gpu));
    CUDA_CHK(cudaFree(img_gpu_out));

}

void main_blur_cpu(float *img_cpu, float *img_cpu_out, int width, int heigth, int k){
    MS(blur_cpu(img_cpu, width, heigth, img_cpu_out, k), time);

    PRINT_CSV("blur_cpu", width, heigth, time)
}



