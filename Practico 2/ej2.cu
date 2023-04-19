#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "math.h"


#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void read_file(const char*, int*);
int get_text_length(const char * fname);

#define N 512
#define CHAR_SIZE 256

__device__ int modulo(int a, int b){
	int r = a % b;
	r = (r < 0) ? r + b : r;
	return r;
}

__global__ void count_kernel(int *d_message, int *d_vector, int length)
{
    int pos =  blockDim.x * blockIdx.x + threadIdx.x;
	if (pos < length){
        int letra = d_message[pos];
        atomicAdd(&d_vector[letra], 1);
    }
}

int main(int argc, char *argv[])
{
	
	int *h_message, *h_vector;
	int *d_message, *d_vector;
    
	unsigned int size;
    unsigned int sizeVector = CHAR_SIZE*sizeof(int);

	const char * fname;

	if (argc < 2) printf("Debe ingresar el nombre del archivo\n");
	else
		fname = argv[1];

	int length = get_text_length(fname);

	size = length * sizeof(int);

	// reservar memoria para el mensaje
	h_message = (int *)malloc(size);
    h_vector  = (int *)malloc(sizeVector);

	// leo el archivo de la entrada
	read_file(fname, h_message);

	/* reservar memoria en la GPU */
	CUDA_CHK ( cudaMalloc((void**)& d_message, size) ); // -- Aloca la memoria de la GPU
    CUDA_CHK ( cudaMalloc((void**)& d_vector,  sizeVector) ); // -- Aloca la memoria de la GPU

	/* copiar los datos de entrada a la GPU */
	CUDA_CHK ( cudaMemcpy(d_message, h_message, size, cudaMemcpyHostToDevice) );
    cudaMemset(d_vector, 0, sizeVector);

	/* Configurar la grilla y lanzar el kernel */
	dim3 gridSize (ceil((double)length/N), 1);
	dim3 blockSize(N, 1, 1);

	count_kernel<<<gridSize, blockSize>>>(d_message, d_vector, length);
	CUDA_CHK( cudaGetLastError() )

	/* Copiar los datos de salida a la CPU en h_message */
	CUDA_CHK(cudaMemcpy(h_vector, d_vector, sizeVector, cudaMemcpyDeviceToHost));
	
    int total = 0;
	// despliego el mensaje
	for (int i = 0; i < CHAR_SIZE; i++) {
        total = total + h_vector[i];
		printf("Caracter (Nº %d) '%c' ocurre: %d veces\n", i , (char)i, h_vector[i]);
	}
	printf("El total es %d\n" , total);

	// libero la memoria en la GPU
	CUDA_CHK(cudaFree(d_message));
    CUDA_CHK(cudaFree(d_vector));

	// libero la memoria en la CPU
	free(h_message);
    free(h_vector);

	return 0;
}

	
int get_text_length(const char * fname)
{
	FILE *f = NULL;
	f = fopen(fname, "r"); //read and binary flags

	size_t pos = ftell(f);    
	fseek(f, 0, SEEK_END);    
	size_t length = ftell(f); 
	fseek(f, pos, SEEK_SET);  

	fclose(f);

	return length;
}

void read_file(const char * fname, int* input)
{
	// printf("leyendo archivo %s\n", fname );

	FILE *f = NULL;
	f = fopen(fname, "r"); //read and binary flags
	if (f == NULL){
		fprintf(stderr, "Error: Could not find %s file \n", fname);
		exit(1);
	}

	//fread(input, 1, N, f);
	int c; 
	while ((c = getc(f)) != EOF) {
		*(input++) = c;
	}

	fclose(f);
}
