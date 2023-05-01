
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



// __global__ void blur_kernel(float* d_input, float* d_output, float* d_msk, int width, int height){

// }

__global__ void ajustar_brillo_coalesced_kernel(float* o_img, float* d_img, int width, int height, float coef){
    
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;

    int pos = pos_x + pos_y * width;

    if (pos_x < width && pos_y < height)
        d_img[pos] = min(255.0f,max(0.0f,o_img[pos]+coef));

}

// __global__ void ajustar_brillo_no_coalesced_kernel(float* d_img, int width, int height, float coef){

// }


// void blur_gpu(float * img_in, int width, int height, float * img_out, float msk[], int m_size){
    
// }

void ajustar_brillo_cpu(float * img_in, int width, int height, float * img_out, float coef){

    for(int imgx=0; imgx < width ; imgx++){
        for(int imgy=0; imgy < height; imgy++){
            img_out[imgy*width+imgx] = std::min(255.0f,std::max(0.0f,img_in[imgy*width+imgx]+coef));
        }
    }
    
}


void ejecutar_kernel_ajuste_brillo_coalesced_y_tomar_tiempo(dim3 gridSize, dim3 bloakSize, float *img_gpu, float *img_gpu_out, int width, int heigth, float coeficiente){
    ajustar_brillo_coalesced_kernel<<<gridSize,bloakSize >>>(img_gpu, img_gpu_out, width, heigth, coeficiente );
    CUDA_CHK( cudaGetLastError() );

    CUDA_CHK(cudaDeviceSynchronize())


}

void main_ajuste_brillo_cpu(float * img_in, int width, int height, float * img_out, float coef){
    MS(ajustar_brillo_cpu(img_in, width, height, img_out, coef), time);

    printf("El tiempo de ejecucion de ajuste_brillo_cpu es %.4f ms", time);
}


void main_ajuste_brillo_coalesced(float *img_cpu, float *img_cpu_out, int width, int heigth, float coeficiente){

    float * img_gpu = NULL;
    float * img_gpu_out = NULL;

    size_t size = width * heigth * sizeof(float);
    CUDA_CHK ( cudaMalloc((void**)& img_gpu, size) );
    CUDA_CHK ( cudaMalloc((void**)& img_gpu_out, size) );

    CUDA_CHK ( cudaMemcpy(img_gpu, img_cpu, size, cudaMemcpyHostToDevice) );
    CUDA_CHK ( cudaMemcpy(img_gpu_out, img_cpu_out, size, cudaMemcpyHostToDevice) );


    printf("Tam= %d ; %d ; %.2f ; %.2f \n" , width, heigth,  ((float)width)/32 ,  ((float)heigth)/16 );
    dim3 gridSize( (int)((float)width)/32, (int)((float)heigth)/16 );
    dim3 blockSize(32, 16);

	MS(ejecutar_kernel_ajuste_brillo_coalesced_y_tomar_tiempo(gridSize, blockSize, img_gpu, img_gpu_out, width, heigth, coeficiente), time);

	printf("Tiempo de ejecucion ajuste_brillo_coalesced: %.4f ", time);

    CUDA_CHK(cudaMemcpy(img_cpu_out, img_gpu_out, size, cudaMemcpyDeviceToHost));

    CUDA_CHK(cudaFree(img_gpu));
    CUDA_CHK(cudaFree(img_gpu_out));

}




