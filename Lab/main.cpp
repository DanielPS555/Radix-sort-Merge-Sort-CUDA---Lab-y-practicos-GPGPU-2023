
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "include/lab.h"

#include <time.h>

int main(){
    /* initialize random seed: */
    srand (time(NULL));

    int length = 2048;

    int * array32 = (int*)malloc(length * sizeof (int));
    for (int i = 0; i < length; ++i) {
        int x_rand = rand() % 90;
        array32[i] = x_rand;
    }
    printf("Original");
    for (int i = 0; i < length; ++i) {
        if(i % 32 == 0)
            printf("\n");
        printf("%d,", array32[i]);
    }

    order_array(array32, length);

    printf("\nLuego");
    for (int i = 0; i < length; ++i) {
        if(i % 32 == 0)
            printf("\n");
        printf("%d,", array32[i]);
    }

    free(array32);

    return 0;
}