#ifndef CIRCLE_H
#define CIRCLE_H

#include "fields.cuh"
#include "point.cuh"

extern "C"
m31* sort_values_and_permute_with_bit_reverse_order(m31 *from, int size);

extern "C"
m31* precompute_twiddles(point initial, point step, int total_size);

extern "C"
void interpolate(uint32_t *values, uint32_t *inverse_twiddles_tree, int values_size);

#endif // CIRCLE_H