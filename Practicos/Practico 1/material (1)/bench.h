#include <math.h>
#include <time.h>
#include <stdlib.h>

#define VALT double
#define N 512
#define BILLION 1000000000L
#define REPS 10
#define N_MEDIAN 5

#define MS(f,elap)                                                                                           \
        double elap=0;                                                                                       \
        {                                                                                                    \
        struct timespec t_ini,t_fin;                                                                         \
            clock_gettime(CLOCK_MONOTONIC, &t_ini);                                                          \
            f;                                                                                               \
            clock_gettime(CLOCK_MONOTONIC, &t_fin);                                                          \
            elap = 1000 * (t_fin.tv_sec - t_ini.tv_sec) + (t_fin.tv_nsec - t_ini.tv_nsec)/1000000.0;         \
        }

#define PRINT_RESULT(name,t)                                                                                 \
    printf("%s: %.2f ms, %.2f GFlops\n" , name, t , ( ((double)m/t )*((double)n/ 1000.0)*((double)p/1000.0)) );

#define PRINT_RESULT2(name,n,t)                                                                                 \
    printf("%s(%d)= %.2f;\n" , name, n, t);

#define PRINT_RESULT_BLK2(name,n,nb,t)                                                                                 \
    printf("%s(%d,%d)= %.2f;\n" , name, n, nb, t);
