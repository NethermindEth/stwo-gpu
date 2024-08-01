#include "../include/circle.cuh"
#include "../include/batch_inverse.cuh"
#include "../include/bit_reverse.cuh"
#include "../include/fields.cuh"
#include "../include/point.cuh"
#include "../include/utils.cuh"
#include <assert.h>
#include <stdint.h>
#include <cstdio>

uint32_t* accumulate_quotients(
    size_t domain_initial_index,
    size_t domain_step_size,
    uint32_t domain_log_size, 
    uint32_t** columns, 
    size_t columns_size,
    size_t columns_row_size,
    QM31 random_coeff
    // ColumnSampleBatch* sample_batches,
    // size_t sample_batches_size 
    ) {
    
    // Initialize Structs
    M31** new_columns = reinterpret_cast<M31**>(columns);
    CircleDomain domain = CircleDomain(Coset(CirclePointIndex(domain_initial_index), CirclePointIndex(domain_step_size), domain_log_size)); 

    // Dev Testing Print Loop
    // for(int i = 0; i < columns_size; i++) {
    //     uint32_t* host_columns = (uint32_t*)malloc(columns_row_size * sizeof(uint32_t));
    //     cudaMemcpy(host_columns, new_columns[i], sizeof(uint32_t) * columns_row_size, cudaMemcpyDeviceToHost);
    //     for(int j = 0; j < columns_row_size; j++) {
    //         printf("M31(%d), ", host_columns[j]);
    //     }
    //     printf("\n");
    // }

    // int threadsPerBlock = 1024;  // This can be adjusted based on your GPU and needs
    // int blocksPerGrid = (sample_batches_size + threadsPerBlock - 1) / threadsPerBlock;
    // accumulate_quotients_helper<<<blocksPerGrid, threadsPerBlock>>>(domain, columns, columns_size, circle_evaluation_size, random_coeff, sample_batches, sample_batches_size);

}

// __global__
// void accumulate_quotients_helper (
//     CircleDomain domain, 
//     CircleEvaluation** columns, 
//     size_t columns_size,
//     size_t circle_evaluation_size,
//     QM31 random_coeff,
//     ColumnSampleBatch* sample_batches ) {
  
//     int row = blockIdx.x * blockDim.x + threadIdx.x;
    
//     CirclePoint<M31> point = domain.at(bit_reverse_index(row, domain.log_size())); 
//     //QM31 row_value = accumulate_row_quotients(sample_batches, columns, )
// }

// __device__ 
// QM31 accumulate_row_quotients(
//     ColumnSampleBatch* sample_batches,
//     CircleEvaluation** columns, 
//     QuotientConstants* quotient_constants,
//     size_t quotient_size,
//     size_t row
// ) {
//     QM31 row_accumulator = QM31::zero(); 
    
//     for(int i = 0; i < quotient_size; ++i) {
//         QM31 numerator = QM31::zero(); 
//         QM31 batch_random_coeffs = quotient_constants->batch_random_coeffs[i]; 
//         QM31* denominator_inverse = quotient_constants->denominator_inverses[i];

//         for(int j = 0; j < sample_batches[i].cv_size; ++j) {
//             size_t column_index = sample_batches[i].columns[j]; 
//             QM31 sampled_value = sample_batches[i].values[j];

//             CircleEvaluation* column = columns[column_index];
//             M31 value = column->values[row]; 
//             numerator = numerator + sub_from_m31(value, sampled_value);
//         }

//         row_accumulator = row_accumulator * batch_random_coeffs + numerator * denominator_inverse[row]; 
//     }
//     return row_accumulator;
// }

// __device__ void
// packed_point_vanishing_fraction(CirclePoint<QM31> vanish_point, CirclePoint<QM31> p, QM31* out1, QM31* out2) {
//     CirclePoint<QM31> e_conjugate = vanish_point.conjugate(); 
//     QM31 h_x = e_conjugate.x * p0 - e_conjugate.x * p1;
//     QM31 h_y = e_conjugate.y * p0 - e_conjugate.y * p1;

//     *out1 = h_y;
//     *out2 = QM31::one() + h_x; 
// }