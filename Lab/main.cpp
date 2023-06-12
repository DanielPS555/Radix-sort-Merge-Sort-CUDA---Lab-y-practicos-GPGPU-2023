
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "include/scan.h"

int main(){


    int * array32 = (int*)malloc(32 * sizeof (int));
    for (int i = 0; i < 32; ++i) {
        array32[i] = 1;
    }

    int * array32Dst = (int*) malloc(32 * sizeof(int));
    pruebaScan(array32, array32Dst);

    for (int i = 0; i < 32; ++i) {
        printf("%d,", array32Dst[i]);
    }

    free(array32);
    free(array32Dst);

    return 0;
}