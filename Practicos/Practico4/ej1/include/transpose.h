//
// Created by bruno on 22/05/23.
//

#ifndef PRACTICO4_TRANSPOSE_H
#define PRACTICO4_TRANSPOSE_H

#define BLOCK_SIZE 32

enum algorithm_type {
    SIMPLE_TRANSPOSE,
    IMPROVED_TRANSPOSE,
    IMPROVED_TRANSPOSE_DUMMY,
};

double execute_kernel(algorithm_type algo, float* in_cpu_m, float* out_cpu_m, int width, int height);

#endif //PRACTICO4_TRANSPOSE_H
