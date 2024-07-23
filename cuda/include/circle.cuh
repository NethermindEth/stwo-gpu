#ifndef CIRCLE_H
#define CIRCLE_H

#include "fields.cuh"
#include "point.cuh"

extern "C"
m31* sort_values_and_permute_with_bit_reverse_order(m31 *from, int size);

extern "C"
m31* precompute_twiddles(point initial, point step, int total_size);

extern "C"
void interpolate(m31 *values, m31 *inverse_twiddles_tree, int values_size);

extern "C"
void evaluate(m31 *values, m31 *inverse_twiddles_tree, int values_size);

extern "C"
qm31 eval_at_point(m31 *coeffs, int coeffs_size, qm31 point_x, qm31 point_y);

__device__ int get_twiddle(m31 *twiddles, int index);

#endif // CIRCLE_H