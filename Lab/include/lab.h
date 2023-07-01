//
// Created by bruno on 13/06/23.
//

#ifndef LAB_LAB_H
#define LAB_LAB_H

#include "utils.h"

void test_radix_sort(int * src);

void test_with_block_under_256(int * srcCpu, int length);
void test_secuence_reading (int * srcCpu, int length);
void order_array (int * srcCpu, int length);

void test_merge_segment_using_separators(int * array, int largo, int * sa, int * sb, int maximoSoporadoPorMergeSort,int numeroPartes);

//Este metodo es posta, no borrar
void order_with_trust(int * src, int length);

#endif //LAB_LAB_H
