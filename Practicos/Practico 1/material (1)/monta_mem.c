#include <stdio.h>
#include "bench.h"
#include "stdint.h"
#include <time.h>
#include <math.h>

void test(int * data, int elems, int stride, int reps) {
	int i,r, result = 0;
	volatile int sink;

	// si reps es muy bajo el overhead de la función clock_gettime incide en los resultados
	for (int r = 0; r < reps; ++r)
	for (i = 0; i < elems; i += stride)
		result += data[i];
	sink = result;
}

int compare(const void *a, const void *b) {
    return (*(uint64_t *)a - *(uint64_t *)b);
}

float get_median(uint64_t arr[]) {
    qsort(arr, N_MEDIAN, sizeof(uint64_t), compare);
    if (N_MEDIAN % 2 == 0) {
        return (float)(arr[N_MEDIAN / 2] + arr[(N_MEDIAN / 2) - 1]) / 2;
    } else {
        return (float)arr[N_MEDIAN / 2];
    }
}

// ejecuta el test N_MEDIAN veces y devuelve la mediana del tiempo de ejecución en nanosegundos
uint64_t run(int * data, int size, int stride)
{

	uint64_t diff[N_MEDIAN];
	struct timespec t_ini,t_fin;                                                                         

	int elems = size / sizeof(int);
	test(data, elems, stride,REPS); /* warm up the cache */

	int suma=0;

	// se toma la mediana de varias corridas para evitar outliers causados por ejecutar en un entorno compartido.
	for (int i = 0; i < N_MEDIAN; ++i)
	{
		clock_gettime(CLOCK_MONOTONIC, &t_ini);
		test(data, elems, stride,REPS);
		clock_gettime(CLOCK_MONOTONIC, &t_fin);

		diff[i] = (BILLION * (t_fin.tv_sec - t_ini.tv_sec) + (t_fin.tv_nsec - t_ini.tv_nsec))/REPS;
	}
	
	return floor(get_median(diff));
}


int main(char argc, char * argv[]){

    if (argc < 3) {
        printf("El programa recibe el tamaño máximo del arreglo en MB y el stride máximo\n");
        exit(1);
    }

    const uint MINBYTES = 16*1024; // comienza de 16KB
    uint MAXBYTES = atoi(argv[1]) * 1024 *1024;
    int MAXSTRIDE = atoi(argv[2]);
	int MAXELEMS=MAXBYTES/sizeof(int);

	int * data = (int *) aligned_alloc(64,MAXELEMS*sizeof(int));
 
	int size; 
	int stride;
	int i;

	//inicializo el arreglo de datos...
	for (i = 0; i < MAXELEMS; i++) { data[i]=1;}

	int sz_tam = 1+ceil(log(MAXBYTES/MINBYTES)/log(2));

	printf("tam = zeros(1,%d);\n", sz_tam );
	printf("str = zeros(1,%d);\n", MAXSTRIDE );
	printf("thr = zeros(%d,%d);\n", sz_tam, MAXSTRIDE );

	i = 1;
	for (size = MAXBYTES; size >= MINBYTES; size >>= 1) {
		printf("tam(%d)=%d;\n", i, size/1024);
		i++;
	}

	i = 1;
	for (stride = 1; stride <= MAXSTRIDE; stride++){
		printf("str(%d)=%d;\n", i, stride);
		i++;
	}

	// se ejecuta el test y los resultados están en GB/s (bytes/ns)
	i = 1;
	for (size = MAXBYTES; size >= MINBYTES; size >>= 1) {
		int j = 1;
		for (stride = 1; stride <= MAXSTRIDE; stride++){
			uint64_t ns=run(data, size, stride);
			printf("%f\t%d\t%d\n",(double)(size/stride)/(double)ns, size, stride);
			j++;
		}
		i++;
	}
	
	return 0;
}