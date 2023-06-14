//
// Created by bruno on 13/06/23.
//

#ifndef LAB_LAB_H
#define LAB_LAB_H

#include "utils.h"

void pruebaScan(int * srcCpu, int * dstCpu);

__device__
void exlusiveScan(int * src, int *dst, int posInicioSrc, int posInicioDst);

/**
 * Order the array based on the byte mask.
 * @param array The array to be ordered (in shared memory). Use pointer arithmetic to privatize the array.
 * @param temp_array The temporary array to be used (in shared memory). Use pointer arithmetic to privatize the array.
 */
void split(int* array, bool* prefix_array, int mask);

/**
 * Test the radix sort.
 * @param srcCpu
 * @param dstCpu
 */
void test_radix(int * srcCpu, int * dstCpu);



#endif //LAB_LAB_H
