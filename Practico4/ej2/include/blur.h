//
// Created by dpadron on 23/05/23.
//

#ifndef PRACTICO4_BLUR_H
#define PRACTICO4_BLUR_H

enum algorithm_type {
    BLUR_WITH_SHARED
    BLUR_WITHOUT_SHARED
};


double execute_kernel(algorithm_type algo, float* in_cpu_m, float* out_cpu_m, int width, int height, int k);

#endif //PRACTICO4_BLUR_H


