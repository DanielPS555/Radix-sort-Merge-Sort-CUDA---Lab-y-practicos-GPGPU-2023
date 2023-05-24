//
// Created by bruno on 22/05/23.
//

#ifndef PRACTICO4_HISTOGRAM_H
#define PRACTICO4_HISTOGRAM_H

#define COLOR_SIZE 256

enum algorithm_type {
    SIMPLE_HISTOGRAM,
    SHARED_MEMORY_HISTOGRAM,
    IMPROVED_SHARED_MEMORY_HISTOGRAM,
};

double execute_kernel(algorithm_type algo, float* in_cpu_m, float* out_cpu_m, int width, int height);

#endif //PRACTICO4_HISTOGRAM_H
