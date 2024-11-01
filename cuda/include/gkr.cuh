#ifndef GKR_H
#define GKR_H

#include "fields.cuh"

template <typename T>
struct Fraction{
    T numerator;
    qm31 denominator;
};

__host__ __device__ Fraction<qm31> add_fraction(Fraction<m31> lhs, Fraction<qm31> rhs);
__host__ __device__ Fraction<qm31> add_fraction(Fraction<qm31> lhs, Fraction<qm31> rhs);

extern "C"
void gen_eq_evals(qm31 v, qm31 *y, uint32_t y_size, qm31 *evals, uint32_t evals_size);

void next_grand_product_layer(qm31 *layer, uint32_t layer_size, qm31 *next_layer, uint32_t next_layer_size);

void next_logup_generic_layer(
    qm31 *numerators, uint32_t numerators_size, qm31 *denominators, uint32_t denominators_size, 
    qm31 *next_numerators, uint32_t next_numerators_size, qm31 *next_denominators, uint32_t next_denominators_size
);

#endif // GKR_H
