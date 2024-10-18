#ifndef POLY_EVAL_AT_POINT_H
#define POLY_EVAL_AT_POINT_H

#include "../fields.cuh"

extern "C"
qm31 eval_at_point(m31 *coeffs, int coeffs_size, qm31 point_x, qm31 point_y);

extern "C"
void evaluate_polynomials_out_of_domain(
    qm31 **result, m31 **polynomials, int *polynomial_sizes, int number_of_polynomials,
    qm31 **out_of_domain_points_x, qm31 **out_of_domain_points_y, int *sample_sizes
);

#endif // POLY_EVAL_AT_POINT_H