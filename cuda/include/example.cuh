#ifndef EXAMPLE_H
#define EXAMPLE_H

#include "fields.cuh"
#include "point.cuh"

extern "C"
void fibonacci_component_evaluate_constraint_quotients_on_domain(
    m31 *evals,
    int evals_size,
    m31 *output_0,
    m31 *output_1,
    m31 *output_2,
    m31 *output_3,
    m31 claim_value,
    point initial_point,
    point step_point,
    qm31 random_coeff_0,
    qm31 random_coeff_1
);

#endif // EXAMPLE_H