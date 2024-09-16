#ifndef POLY_TWIDDLES_H
#define POLY_TWIDDLES_H

#include "fields.cuh"
#include "point.cuh"

extern "C"
void sort_values_and_permute_with_bit_reverse_order(m31 *from, m31 *dst, int size);

extern "C"
void precompute_twiddles(m31* result, point initial, point step, int size);

#endif // POLY_TWIDDLES_H