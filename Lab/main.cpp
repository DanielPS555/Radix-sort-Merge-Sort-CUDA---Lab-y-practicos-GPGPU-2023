
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "include/lab.h"

#include <time.h>



void pruebaKernelD2() {

    int length = 16;
    int t = 4;

    int a[32] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53,
                 55, 57, 59, 61, 63};
    int b[32] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54,
                 56, 58, 60, 62, 64};

    //Sa = {1,33,63}
    //Sb = {2,34,64}

    //S  = {1,2,33,34,63,64}

    int Pa [18] = {0,1,4,5, 8  ,9,12,13,16,17,20,21,24,25,28,29,31,32};
    int Pb [18] = {0,0,4,4, 8  ,8,12,12,16,16,20,20,24,24,28,28,31,31};

    int * array = (int*) malloc(64*sizeof (int));


    int * Pa_dinamic = (int*) malloc(18*sizeof (int));
    int * Pb_dinamic = (int*) malloc(18*sizeof (int));

    for (int i = 0; i < length / 2; ++i) {
        array[i] = a[i];
        array[32 + i] = b[i];
    }

    for (int i = 0; i < length / t + 2; ++i) {
        Pa_dinamic[i] = Pa[i];
        Pb_dinamic[i] = Pb[i];
    }

    printf("\nOriginal");
    for (int i = 0; i < length; ++i) {
        if(i % 32 == 0)
            printf("\n");
        printf("%d,", array[i]);
    }


    test_merge_segment_using_separators(array, length, Pa_dinamic, Pb_dinamic, t*2);

    printf("\nResultado");
    for (int i = 0; i < length; ++i) {
        if(i % 32 == 0)
            printf("\n");
        printf("%d,", array[i]);
    }

    free(array);
    free(Pa_dinamic);
    free(Pb_dinamic);
}




int main(){
    /* initialize random seed: */
    pruebaKernelD2();
    return 0;


    srand (time(NULL));

    int length = 256;

    int * array32 = (int*)malloc(length * sizeof (int));
    for (int i = 0; i < length; ++i) {
        int x_rand = rand() % 20;
        array32[i] = x_rand;
    }
    printf("Original");
    for (int i = 0; i < length; ++i) {
        if(i % 32 == 0)
            printf("\n");
        printf("%d,", array32[i]);
    }

    // pruebaScan(array32, array32Dst);
    test_with_block_under_256(array32, length);
    printf("Luego");
    for (int i = 0; i < length; ++i) {
        if(i % 32 == 0)
            printf("\n");
        printf("%d,", array32[i]);
    }

    free(array32);

    return 0;
}



