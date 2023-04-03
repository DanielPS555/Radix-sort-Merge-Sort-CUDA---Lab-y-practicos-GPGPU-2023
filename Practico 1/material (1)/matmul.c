#include <stdio.h>
#include "bench.h"
#include "stdint.h"
#include <time.h>
#include <math.h>

int mult_simple (const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, int m, int n, int p); 
int mult_fila   (const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, int m, int n, int p); 
int mult_bloque3(const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, int m, int n, int p, int nb); 
int mult_bloque4(const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, int m, int n, int p, int nb); 

void init_vector(VALT *a, size_t n, float val) {
    for (unsigned int i = 0; i < n; i++) a[i] = val;
}

int mult_ijk(const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, int m, int n, int p){
	VALT sum;
	for (int row = 0; row < m; row++)
		for (int col = 0; col < n; col++){
			sum=0;
			for (int it = 0; it < p; it++)
				sum += A[row * p + it] * B[it * n + col];
			C[row * n + col]=sum;
		}
} 

int mult_jik(const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, int m, int n, int p){
	VALT sum;

	for (int col = 0; col < n; col++)
		for (int row = 0; row < m; row++){
			sum=0;
			for (int it = 0; it < p; it++)
				sum += A[row * p + it] * B[it * n + col];
			C[row * n + col]=sum;
		}
}

int mult_ikj(const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, int m, int n, int p){

	VALT a;
	for (int row = 0; row < m; row++)
		for (int it = 0; it < p; it++){
			a=A[row * p + it];
			for (int col = 0; col < n; col++)
				C[row * n + col] += a * B[it * n + col];
		}
} 


int mult_kij(const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, int m, int n, int p){

	VALT a;
	for (int it = 0; it < p; it++)
		for (int row = 0; row < m; row++){
			a=A[row * p + it];
			for (int col = 0; col < n; col++)
				C[row * n + col] += a * B[it * n + col];
		}

} 

int mult_jki(const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, int m, int n, int p){

	VALT b;
	for (int col = 0; col < n; col++)
		for (int it = 0; it < p; it++){
			b=B[it * n + col];
			for (int row = 0; row < m; row++)
				C[row * n + col] += A[row * p + it] * b;
		}

} 

int mult_kji(const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, int m, int n, int p){

	VALT b;
	for (int col = 0; col < n; col++)
		for (int it = 0; it < p; it++){
			b=B[it * n + col];
			for (int row = 0; row < m; row++)
				C[row * n + col] += A[row * p + it] * b;
		}

} 

int mult_bikj(const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, int m, int n, int p, int nb){

	VALT a;

	for (int col_bl = 0; col_bl < n; col_bl += nb)
		for (int row_bl = 0; row_bl < m; row_bl+=nb)
			for (int it_bl = 0; it_bl < p; it_bl += nb)
				for (int row = row_bl; row < row_bl+nb; row++)
					for (int it = it_bl; it < it_bl+nb; it++){
						a=A[(row) * p + it];
						for (int col = col_bl; col < col_bl+nb; col++)
							C[(row) * n + col] += a * B[(it) * n + col];
					}
}

int mult_bijk(const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, int m, int n, int p, int nb){

	VALT sum;
	for (int col_bl = 0; col_bl < n; col_bl += nb)
		for (int row_bl = 0; row_bl < m; row_bl+=nb)
			for (int it_bl = 0; it_bl < p; it_bl += nb)
				for (int row = row_bl; row < row_bl+nb; row++)
					for (int col = col_bl; col < col_bl+nb; col++){
						sum=0;
						for (int it = it_bl; it < it_bl+nb; it++)
							sum += A[(row) * p + it] * B[(it) * n + col];
						C[(row) * n + col] = sum;
					}
}

int main(char argc, char * argv[]){

    // const char * fname;

    if (argc == 4) {

	    int m = atoi(argv[1]);
	    int n = atoi(argv[2]);
	    int p = atoi(argv[3]);
	    int nb = atoi(argv[4]);

	    srand(0); // Inicializa la semilla aleatoria

	    VALT * A = (VALT *) aligned_alloc( 64, m*p*sizeof(VALT) ); // Almacena los elementos en la diagonal de una matriz.
	    VALT * B = (VALT *) aligned_alloc( 64, p*n*sizeof(VALT) ); // Almacena los elementos en la diagonal de una matriz.
	    VALT * C = (VALT *) aligned_alloc( 64, m*n*sizeof(VALT) ); // Almacena los elementos en la diagonal de una matriz.

	    init_vector(A, m*p, 1);
	    init_vector(B, p*n, 1);
	    init_vector(C, m*n, 0);

	    MS( mult_ijk(A,B,C,m,n,p)    , t_mm_ijk  )
	    MS( mult_jik(A,B,C,m,n,p)    , t_mm_jik  )
	    MS( mult_ikj(A,B,C,m,n,p)    , t_mm_ikj  )
	    MS( mult_kij(A,B,C,m,n,p)    , t_mm_kij  )
	    MS( mult_jki(A,B,C,m,n,p)    , t_mm_jki  )
	    MS( mult_kji(A,B,C,m,n,p)    , t_mm_kji  )
	    MS( mult_bijk(A,B,C,m,n,p,nb), t_mm_bijk )
	    MS( mult_bikj(A,B,C,m,n,p,nb), t_mm_bikj )

	    printf("Lineales:\n");
	    PRINT_RESULT("ijk: ", t_mm_ijk )
	    PRINT_RESULT("jik: ", t_mm_jik )
	    PRINT_RESULT("ikj: ", t_mm_ikj )
	    PRINT_RESULT("kij: ", t_mm_kij )
	    PRINT_RESULT("jki: ", t_mm_jki )
	    PRINT_RESULT("kji: ", t_mm_kji )

	    printf("Por bloques:\n");
	    PRINT_RESULT("bijk:", t_mm_bijk )
	    PRINT_RESULT("bikj:", t_mm_bikj )

    }else if (argc == 3) {
    	
	    int MAXN  = atoi(argv[1]);;
	    int MAXNB = atoi(argv[2]); ;

	    VALT * A = (VALT *) aligned_alloc( 64, MAXN*MAXN*sizeof(VALT) ); // Almacena los elementos en la diagonal de una matriz.
	    VALT * B = (VALT *) aligned_alloc( 64, MAXN*MAXN*sizeof(VALT) ); // Almacena los elementos en la diagonal de una matriz.
	    VALT * C = (VALT *) aligned_alloc( 64, MAXN*MAXN*sizeof(VALT) ); // Almacena los elementos en la diagonal de una matriz.

	    init_vector(A, MAXN*MAXN, 1);
	    init_vector(B, MAXN*MAXN, 1);
	    init_vector(C, MAXN*MAXN, 0);

	    printf("ijk, jik, ikj, kij, jki, kji, bijk8, bijk16, bijk32, bijk64, bijk128, bikj8, bikj16, bikj32, bikj64, bikj128\n");


	    for (int n = 128; n <= MAXN; n+=128)
	    {
		    MS( mult_ijk(A,B,C,n,n,n)    , t_mm_ijk  )
		    MS( mult_jik(A,B,C,n,n,n)    , t_mm_jik  )
		    MS( mult_ikj(A,B,C,n,n,n)    , t_mm_ikj  )
		    MS( mult_kij(A,B,C,n,n,n)    , t_mm_kij  )
		    MS( mult_jki(A,B,C,n,n,n)    , t_mm_jki  )
		    MS( mult_kji(A,B,C,n,n,n)    , t_mm_kji  )

		    printf("%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,",t_mm_ijk,t_mm_jik,t_mm_ikj,t_mm_kij,t_mm_jki,t_mm_kji);


		    for (int nb = 8; nb <= MAXNB; nb*=2)
		    {
			    MS( mult_bijk(A,B,C,n,n,n,nb), t_mm_bijk )
				printf("%.2f,",t_mm_bijk);
		    }

		    for (int nb = 8; nb <= MAXNB; nb*=2)
		    {
			    MS( mult_bikj(A,B,C,n,n,n,nb), t_mm_bikj )
				printf("%.2f,",t_mm_bikj);
		    }

		    printf("\n");
	    }
    }else{
    	printf("Ojo con los argumentos...\n");
    }

	return 0;
}