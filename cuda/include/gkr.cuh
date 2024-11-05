#ifndef GKR_H
#define GKR_H

#include "fields.cuh"

extern "C"
void gen_eq_evals(qm31 v, qm31 *y, uint32_t y_size, qm31 *evals, uint32_t evals_size);

template <typename T>
struct Fraction{
    T numerator;
    qm31 denominator;

    __device__ Fraction() : numerator(T{}), denominator(qm31{}) {}
    __device__ Fraction(T num, qm31 denom) : numerator(num), denominator(denom) {}
};

__device__ Fraction<qm31> add_fraction(Fraction<m31> lhs, Fraction<qm31> rhs);
__device__ Fraction<qm31> add_fraction(Fraction<qm31> lhs, Fraction<qm31> rhs);

template <typename T>
struct Reciprocal{
    T x;
    
    __device__ Reciprocal() : x(T{}) {}
    __device__ Reciprocal(T x) : x(x) {}
};

__device__ Fraction<qm31> add_reciprocal(Reciprocal<qm31> lhs, Reciprocal<qm31> rhs);

extern "C" {
    void next_grand_product_layer(
        qm31 *layer, 
        uint32_t layer_size, 
        qm31 *next_layer, 
        uint32_t next_layer_size
    );

    void next_logup_generic_layer(
        qm31 *numerators, 
        qm31 *denominators, 
        uint32_t size, 
        qm31 *next_numerators, 
        qm31 *next_denominators, 
        uint32_t next_size
    );

    void next_logup_multiplicities_layer(
        m31 *numerators, 
        qm31 *denominators, 
        uint32_t size, 
        qm31 *next_numerators, 
        qm31 *next_denominators, 
        uint32_t next_size
    );

    void next_logup_singles_layer(
        qm31 *denominators, 
        uint32_t size, 
        qm31 *next_numerators, 
        qm31 *next_denominators, 
        uint32_t next_size
    );

    void eval_grand_product_sum(
        qm31 *eq_evals,
        qm31 *input_layer, 
        uint32_t n_terms,
        qm31 *eval_at_0,
        qm31 *eval_at_2
    );

    void eval_logup_generic_sum(
        qm31 *eq_evals,
        qm31 *numerators,
        qm31 *denominators,
        uint32_t n_terms,
        qm31 lambda,
        qm31 *eval_at_0,
        qm31 *eval_at_2
    );

    void eval_logup_multiplicities_sum(
        qm31 *eq_evals,
        m31 *numerators,
        qm31 *denominators,
        uint32_t n_terms,
        qm31 lambda,
        qm31 *eval_at_0,
        qm31 *eval_at_2
    );

    void eval_logup_singles_sum(
        qm31 *eq_evals,
        qm31 *denominators,
        uint32_t n_terms,
        qm31 lambda,
        qm31 *eval_at_0,
        qm31 *eval_at_2
    );
}

#endif // GKR_H
