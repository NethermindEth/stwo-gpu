#ifndef POLY_UTILS_H
#define POLY_UTILS_H

#include "fields.cuh"

__device__ int get_twiddle(m31 *twiddles, int index);

__global__ void rescale(m31 *values, int size, m31 factor);

#endif // POLY_UTILS_H
