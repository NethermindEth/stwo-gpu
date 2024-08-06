#ifndef QUOTIENT_H
#define QUOTIENT_H

#include "../include/circle.cuh"
#include "../include/fields.cuh"

#include <assert.h>
#include <stdint.h>

struct SampleColumn {
    size_t usize;
    QM31 secure_field; 
};

struct ColumnSampleBatch {
    CirclePoint<QM31> point;
    size_t* columns; 
    QM31* values; 
    size_t size; 
};

extern "C" 
uint32_t* accumulate_quotients(
    uint32_t* value_columns1,
    uint32_t* value_columns2,
    uint32_t* value_columns3,
    uint32_t* value_columns4,
    size_t domain_initial_index,
    size_t domain_step_size,
    uint32_t domain_log_size, 
    size_t domain_size, 
    uint32_t** columns, 
    size_t columns_size,
    size_t columns_row_size,
    QM31 random_coeff,
    ColumnSampleBatch* sample_batches,
    size_t sample_batches_size
);

#endif