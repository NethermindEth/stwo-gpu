#ifndef QUOTIENTS_H
#define QUOTIENTS_H

#include "fields.cuh"
#include "point.cuh"
#include "utils.cuh"

const unsigned int BLOCK_SIZE = 1024;

extern "C"
void accumulate_quotients(
        uint32_t half_coset_initial_index,
        uint32_t half_coset_step_size,
        uint32_t domain_size,
        m31 **columns,
        uint32_t number_of_columns,
        qm31 random_coefficient,
        secure_field_point *sample_points,
        uint32_t *sample_column_indexes,
        uint32_t sample_column_indexes_size,
        qm31 *sample_column_values,
        uint32_t *sample_column_and_values_sizes,
        uint32_t sample_size,
        uint32_t *result_column_0,
        uint32_t *result_column_1,
        uint32_t *result_column_2,
        uint32_t *result_column_3,
        qm31 *flattened_line_coeffs,
        uint32_t flattened_line_coeffs_size,
        uint32_t *line_coeffs_sizes,
        qm31 *batch_random_coeffs
);

#endif // QUOTIENTS_H