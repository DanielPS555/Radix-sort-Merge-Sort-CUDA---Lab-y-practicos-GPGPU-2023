#include "include/CImg.h"
#include <stdio.h>
#include "include/histogram.h"

using namespace cimg_library;

void execute_algorithm(algorithm_type algo, float* in_cpu_m, float* out_cpu_m, int width, int height) {
    /*
     * Execute and print time of the algorithm
     */
    double time = execute_kernel(algo, in_cpu_m, out_cpu_m, width, height);
    switch (algo) {
        case SIMPLE_HISTOGRAM: {
            printf("SIMPLE_HISTOGRAM,%.4f\n", time);
            break;
        }
        case SHARED_MEMORY_HISTOGRAM: {
            printf("SHARED_MEMORY_HISTOGRAM,%.4f\n", time);
            break;
        }
        case IMPROVED_SHARED_MEMORY_HISTOGRAM: {
            printf("IMPROVED_SHARED_MEMORY_HISTOGRAM,%.4f\n", time);
            break;
        }
    }
}

int main(int argc, char** argv){
    // Arguments
    // path: path to the image
    // algo: algorithm to execute, by default SIMPLE_HISTOGRAM

    if (argc < 2) {
        printf("Debe ingresar el nombre del archivo\n");
        return 0;
    }

    const char * path = argv[1];
    algorithm_type algo = argc > 2 ? (algorithm_type)atoi(argv[2]) : SIMPLE_HISTOGRAM;

    // Read image
    CImg<float> image(path);
    float* cpu_histogram = new float[COLOR_SIZE];
    // Image matrices
    float *img_cpu_matrix = image.data();
    // Execute algorithm
    execute_algorithm(algo, img_cpu_matrix, cpu_histogram, image.width(), image.height());

    char fname[30];
    sprintf(fname, "histogram_%d.txt", algo);

    // Create file histogram_%d.txt
    FILE *f = fopen(fname, "w");
    for (int i = 0; i < COLOR_SIZE; i++) {
        fprintf(f, "%d,%.0f\n", i, cpu_histogram[i]);
    }
    fclose(f);

    return 0;
}



