
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "include/lab.h"

#include <time.h>

int main(){
    /* initialize random seed: */
    srand (time(NULL));

    int * array32 = (int*)malloc(64 * sizeof (int));
    for (int i = 0; i < 64; ++i) {
        int x_rand = rand() % 1000;
        array32[i] = x_rand;
    }

    int * array32Dst = (int*) malloc(64 * sizeof(int));
    // pruebaScan(array32, array32Dst);
    test_radix_sort(array32, array32Dst);

    for (int i = 0; i < 64; ++i) {
        if(i % 32 == 0)
            printf("\n");
        printf("%d,", array32Dst[i]);
    }

    free(array32);
    free(array32Dst);

    return 0;
}