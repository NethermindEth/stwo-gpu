#ifndef POLY_IFFT_H
#define POLY_IFFT_H

#include "../fields.cuh"
#include "../utils.cuh"
#include "utils.cuh"

extern "C"
void interpolate(int eval_domain_size, m31 *values, m31 *inverse_twiddles_tree, int inverse_twiddles_size, int values_size);

extern "C"
void interpolate_columns(int eval_domain_size, m31 **values, m31 *inverse_twiddles_tree, int inverse_twiddles_size, int values_size, int number_of_rows);

#endif // POLY_IFFT_H