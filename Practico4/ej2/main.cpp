#include "include/CImg.h"
#include <stdio.h>
#include "include/blur.h"

using namespace cimg_library;

void execute_algorithm(algorithm_type algo, float* in_cpu_m, float* out_cpu_m, int width, int height, int k) {
    /*
     * Execute and print time of the algorithm
     */
    double time = execute_kernel(algo, in_cpu_m, out_cpu_m, width, height, k);
    switch (algo) {
        case BLUR_WITH_SHARED: {
            printf("BLUR_WITH_SHARED,%.4f\n", time);
            break;
        }
        case BLUR_WITHOUT_SHARED: {
            printf("BLUR_WITHOUT_SHARED,%.4f\n", time);
            break;
        }
    }
}

int main(int argc, char** argv){
    // Arguments
    // path: path to the image
    // algo: algorithm to execute, by default SIMPLE_TRANSPOSE

    if (argc < 2) {
        printf("Debe ingresar el nombre del archivo\n");
        return 0;
    }

    const char * path = argv[1];
    algorithm_type algo = argc > 2 ? (algorithm_type)atoi(argv[2]) : BLUR_WITHOUT_SHARED;

    // Read image
    CImg<float> image(path);
    CImg<float> image_out(image.width(), image.height(),1,1,0);
    // Image matrices
    float *img_cpu_matrix = image.data();
    float *img_cpu_out_matrix = image_out.data();
    // Execute algorithm
    execute_algorithm(algo, img_cpu_matrix, img_cpu_out_matrix, image.width(), image.height(), 10);

    char fname[30];
    sprintf(fname, "output_blur_%d.ppm", algo);
    image_out.save(fname);

    return 0;
}