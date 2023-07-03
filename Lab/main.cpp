#include <stdio.h>
#include "include/lab.h"

#include <time.h>
#include <algorithm>




int * create_random_data(int length) {
    int * data = (int*)malloc(length * sizeof (int));
    for (int i = 0; i < length; ++i) {
        int x_rand = rand() % 100;
        data[i] = x_rand;
    }
    return data;
}


int main() {
    srand (time(NULL));
    FILE * output = fopen("output.txt", "w");

    fprintf(output, "Size,Time,Algorithm\n");

    // Ignore first run
    for (int size = 256; size <= 65536; size *= 2) {
        int * our_data = create_random_data(size);
        int * trust_data = create_random_data(size);
        int * cpu_data = create_random_data(size);

        MS( order_array(our_data, size), our_time )
        MS( order_with_trust(trust_data, size), timeTrust)
        MS( std::sort(cpu_data, cpu_data + size), cpu_time)

        free(our_data);
        free(trust_data);
    }

    for (int size = 256; size <= 65536; size *= 2) {
        int * our_data = create_random_data(size);
        int * trust_data = create_random_data(size);
        int * cpu_data = create_random_data(size);

        MS( order_array(our_data, size), our_time )
        MS( order_with_trust(trust_data, size), timeTrust)
        MS( std::sort(cpu_data, cpu_data + size), cpu_time)

        fprintf(output, "%d,%f,Our\n", size, our_time);
        fprintf(output, "%d,%f,Trust\n", size, timeTrust);
        fprintf(output, "%d,%f,CPU\n", size, cpu_time);

        free(our_data);
        free(trust_data);
    }

    fclose(output);
    return 0;
}
