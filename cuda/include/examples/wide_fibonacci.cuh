#ifndef WIDE_FIBONACCI_H
#define WIDE_FIBONACCI_H

#include "fields.cuh"
#include "utils.cuh"

extern "C"
void evaluate_wide_fibonacci_constraint_quotients_on_domain(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 **trace_evaluations,
    qm31 *random_coeff_powers,
    m31 *denominator_inverses,
    unsigned int extended_domain_size,
    unsigned int number_of_columns
);

#endif // WIDE_FIBONACCI_H