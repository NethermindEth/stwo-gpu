#ifndef QUOTIENTS_H
#define QUOTIENTS_H

#include "fields.cuh"
#include "point.cuh"

const unsigned int BLOCK_SIZE = 1024;

extern "C"
void accumulate_quotients(
        point domain_initial_point,
        point domain_step,
        uint32_t domain_size,
        uint32_t **columns,
        uint32_t number_of_columns,
        qm31 random_coefficient,
        secure_field_point *sample_points,
        uint32_t *sample_column_indexes,
        qm31 *sample_column_values,
        uint32_t *sample_column_and_values_sizes,
        uint32_t sample_size,
        uint32_t *result_column_0,
        uint32_t *result_column_1,
        uint32_t *result_column_2,
        uint32_t *result_column_3
);

#endif // QUOTIENTS_H