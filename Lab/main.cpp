
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "include/lab.h"

#include <time.h>



void pruebaKernelD2() {

    int length = 64;
    int t = 16;

    int b[32] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32};
    int a[32] = {33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,};

    //Sa = {1,17,32}
    //Sb = {33,49,64}

    //S  = {1,17,32,33,49,64}

    int Pb [6] = {0,16,31,32,32,32};
    int Pa [6] = {0,0,0,0,16,31};

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



