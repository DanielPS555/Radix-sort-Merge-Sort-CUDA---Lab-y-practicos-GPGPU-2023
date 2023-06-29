#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "include/lab.h"

#include <time.h>


#define MS(f,elap)                                                                                           \
        double elap=0;                                                                                       \
        {                                                                                                    \
        struct timespec t_ini,t_fin;                                                                         \
            clock_gettime(CLOCK_MONOTONIC, &t_ini);                                                          \
            f;                                                                                               \
            clock_gettime(CLOCK_MONOTONIC, &t_fin);                                                          \
            elap = 1000 * (t_fin.tv_sec - t_ini.tv_sec) + (t_fin.tv_nsec - t_ini.tv_nsec)/1000000.0;         \
        }




int main(){

    int length = 256;


    while (length <= 65536){
        // Notar que para que la matriz de separadores pueda ser ordenable dentro de
        //   un mismo bloque de tamaño no mayor 512, se debe cumplir que los dos array a ordenar no superen los
        //   512 separadores. Como una array de tamaño x tiene x/256 + 1 separadores, se cumple que x no puede ser mayor que 2^16


        int * array = (int*)malloc(length * sizeof (int));
        for (int i = 0; i < length; ++i) {
            int x_rand = rand() % 1000;
            array[i] = x_rand;
        }

        // Se incluyen en la prueba los tiempos de reserva de memoria en la GPU y en la tranferencia, ya que trust administra en su caso
        MS(test_with_trust(array, length), timeTrust)
        //ToDo añadir la llamada a nuestro metodo

        printf("\nSize = %d | Nuestro = ?? | Trust = %.3f ms", length, timeTrust);

        free(array);

        length *= 2;
    }



    return 0;
}