#ifndef POLY_RFFT_H
#define POLY_RFFT_H

#include "../fields.cuh"

extern "C"
void evaluate(int eval_domain_size, m31 *values, m31 *twiddles_tree, int twiddles_size, int values_size);

extern "C"
void evaluate_columns(const int *eval_domain_sizes, m31 **values, m31 *twiddles_tree, int twiddles_size, int number_of_columns, const int *column_sizes);

#endif // POLY_RFFT_H