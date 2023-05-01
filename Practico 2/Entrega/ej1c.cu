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

#define A 15
#define B 27
#define M 256
#define A_MMI_M -17

#define N 512
#define CANT_BLOCK 128


__device__ int modulo(int a, int b){
	int r = a % b;
	r = (r < 0) ? r + b : r;
	return r;
}

__global__ void decrypt_kernel(int *d_message, int lengthPorThread, int length)
{	
	int uid = (blockDim.x * blockIdx.x + threadIdx.x);
	
	int ini = uid * lengthPorThread;
	int last = min(ini + lengthPorThread, length);

	for (int pos = ini; pos < last; pos++){
		d_message[pos] = modulo(A_MMI_M * (d_message[pos] - B),  M);
	}
}

int main(int argc, char *argv[])
{
	int *h_message;
	int *d_message;
	unsigned int size;

	const char * fname;

	if (argc < 2) printf("Debe ingresar el nombre del archivo\n");
	else
		fname = argv[1];

	int length = get_text_length(fname);

	size = length * sizeof(int);

	// reservar memoria para el mensaje
	h_message = (int *)malloc(size);

	// leo el archivo de la entrada
	read_file(fname, h_message);

	/* reservar memoria en la GPU */
	CUDA_CHK ( cudaMalloc((void**)& d_message, size) ); // -- Aloca la memoria de la GPU

	/* copiar los datos de entrada a la GPU */
	CUDA_CHK ( cudaMemcpy(d_message, h_message, size, cudaMemcpyHostToDevice) );

	/* Configurar la grilla y lanzar el kernel */
	dim3 gridSize(CANT_BLOCK, 1);
	dim3 blockSize(N, 1, 1);

	int lenghtPorThread = ceil((float)length / (N * CANT_BLOCK));
	decrypt_kernel<<<gridSize, blockSize>>>(d_message, lenghtPorThread, length);
	CUDA_CHK( cudaGetLastError() );

	/* Copiar los datos de salida a la CPU en h_message */
	CUDA_CHK(cudaMemcpy(h_message, d_message, size, cudaMemcpyDeviceToHost));
	
	// despliego el mensaje
	for (int i = 0; i < length; i++) {
		printf("%c", (char)h_message[i]);
	}
	printf("\n");

	// libero la memoria en la GPU
	CUDA_CHK(cudaFree(d_message));

	// libero la memoria en la CPU
	free(h_message);

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
