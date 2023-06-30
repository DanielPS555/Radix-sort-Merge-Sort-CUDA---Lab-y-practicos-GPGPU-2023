
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "include/lab.h"

#include <time.h>



void pruebaKernelD2() {

    int lengthPorParte = 32;
    int t = 8; //Esto es el numero maximo de elemento que puede tener A o B para hacer el merge sort
    int numeroPartes = 2;

    int arr[64] = {8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
                   7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
                   4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
                   3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,};

    int Pa [12] = {0,0,0, 0, 8, 15,        0,0,0, 0, 8, 15};
    int Pb [12] = {0,8,15,16,16,16,        0,8,15,16,16,16};

    int numeroDeSeparadoresPorParte = (lengthPorParte / (t*2 / 2) + 2);

    int * array = (int*) malloc(lengthPorParte * numeroPartes*sizeof (int));


    int * Pa_dinamic = (int*) malloc(numeroDeSeparadoresPorParte*numeroPartes*sizeof (int));
    int * Pb_dinamic = (int*) malloc(numeroDeSeparadoresPorParte*numeroPartes*sizeof (int));

    for (int i = 0; i < lengthPorParte * numeroPartes; ++i) {
        array[i] = arr[i];
    }

    for (int i = 0; i < numeroDeSeparadoresPorParte * numeroPartes; ++i) {
        Pa_dinamic[i] = Pa[i];
        Pb_dinamic[i] = Pb[i];
    }

    printf("\nNumero de bloques (parejas A-B) = %d", numeroPartes);

    printf("\nOriginal");
    for (int i = 0; i < lengthPorParte * numeroPartes; ++i) {
        if(i % 32 == 0)
            printf("\n");
        printf("%d,", array[i]);
    }


    test_merge_segment_using_separators(array, lengthPorParte, Pa_dinamic, Pb_dinamic, t*2, numeroPartes);

    printf("\nResultado");
    for (int i = 0; i < lengthPorParte * numeroPartes; ++i) {
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



