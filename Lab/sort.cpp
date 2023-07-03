#include <stdio.h>

#include <time.h>
#include <algorithm>

#define MS(f,elap)                                                                                           \
        double elap=0;                                                                                       \
        {                                                                                                    \
        struct timespec t_ini,t_fin;                                                                         \
            clock_gettime(CLOCK_MONOTONIC, &t_ini);                                                          \
            f;                                                                                               \
            clock_gettime(CLOCK_MONOTONIC, &t_fin);                                                          \
            elap = 1000 * (t_fin.tv_sec - t_ini.tv_sec) + (t_fin.tv_nsec - t_ini.tv_nsec)/1000000.0;         \
        }


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
        int * cpu_data = create_random_data(size);

        MS( std::sort(cpu_data, cpu_data + size), cpu_time)

        free(cpu_data);
    }

    for (int size = 256; size <= 65536; size *= 2) {
        int * cpu_data = create_random_data(size);

        MS( std::sort(cpu_data, cpu_data + size), cpu_time)

        fprintf(output, "%d,%f,CPU\n", size, cpu_time);

        free(cpu_data);
    }

    fclose(output);
    return 0;

}