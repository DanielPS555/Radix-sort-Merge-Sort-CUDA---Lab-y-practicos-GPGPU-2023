//
// Created by dpadron on 11/06/23.
//

#ifndef LAB_SCAN_H
#define LAB_SCAN_H

#include "utils.h"


void pruebaScan(int * srcCpu, int * dstCpu);

__device__
void exlusiveScan(int * src, int *dst, int posInicioSrc, int posInicioDst);

#endif //LAB_SCAN_H
