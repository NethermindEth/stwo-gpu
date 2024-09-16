#ifndef POLY_RFFT_H
#define POLY_RFFT_H

#include "../fields.cuh"

extern "C"
void evaluate(int eval_domain_size, m31 *values, m31 *twiddles_tree, int twiddles_size, int values_size);

#endif // POLY_RFFT_H