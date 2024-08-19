#include "../include/quotients.cuh"

#include <cstdio>

#define THREAD_COUNT_MAX 1024 

typedef struct {
    secure_field_point point;
    uint32_t *columns;
    qm31 *values;
    uint32_t size;
} column_sample_batch;

__device__ point index_to_point(uint32_t index) {
    return point_pow(m31_circle_gen, (int)index);
}

__device__ point domain_at_index(uint32_t half_coset_initial_index, uint32_t half_coset_step_size, uint32_t index, uint32_t domain_size) {
    uint32_t half_coset_size = domain_size >> 1;

    if (index < half_coset_size) {
        int modulo_u31_mask = 0x7fffffff;
        uint64_t global_index = (uint64_t) half_coset_initial_index + (uint64_t) half_coset_step_size * (uint64_t) index;
        return index_to_point(global_index & modulo_u31_mask);
    } else {
        int modulo_u31_mask = 0x7fffffff;
        uint64_t global_index = (uint64_t) half_coset_initial_index + (uint64_t) half_coset_step_size * (uint64_t) (index - half_coset_size);
        return index_to_point((2147483648 - global_index) & modulo_u31_mask);
    }
}

void column_sample_batches_for(
        secure_field_point *sample_points,
        uint32_t *sample_column_indexes,
        qm31 *sample_column_values,
        const uint32_t *sample_column_and_values_sizes,
        uint32_t sample_size,
        column_sample_batch *result
) {
    unsigned int offset = 0;
    for (unsigned int index = 0; index < sample_size; index++) {
        result[index].point = sample_points[index];
        result[index].columns = &sample_column_indexes[offset];
        result[index].values = &sample_column_values[offset];
        result[index].size = sample_column_and_values_sizes[index];
        offset += sample_column_and_values_sizes[index];
    }
};

__device__ void complex_conjugate_line_coeffs(secure_field_point point, qm31 value, qm31 alpha, qm31* a_out, qm31* b_out, qm31* c_out) {
    qm31 a = sub(qm31{value.a, neg(value.b)}, value); 
    qm31 c = sub(qm31{point.y.a, neg(point.y.b)}, point.y);
    qm31 b = sub(mul(value, c), mul(a, point.y));  

    *a_out = mul(alpha, a);
    *b_out = mul(alpha, b);
    *c_out = mul(alpha, c);
}

__global__ void column_line_and_batch_random_coeffs(
    column_sample_batch *sample_batches,
    uint32_t sample_size,
    qm31 random_coefficient,
    qm31 *flattened_line_coeffs,
    uint32_t *line_coeffs_sizes,
    qm31 *batch_random_coeffs
) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < sample_size) {
        // Calculate Batch Random Coeffs
        batch_random_coeffs[tid] = pow(random_coefficient, sample_batches[tid].size); 

        // Calculate Column Line Coeffs
        line_coeffs_sizes[tid] = sample_batches[tid].size;
        size_t sample_batches_offset = tid * line_coeffs_sizes[tid] * 3; 

        qm31 alpha = qm31{cm31{m31{1}, m31{0}}, cm31{m31{0}, m31{0}}};

        for(size_t j = 0; j < sample_batches[tid].size; ++j) {
            qm31 sampled_value = sample_batches[tid].values[j];
            alpha = mul(alpha, random_coefficient); 
            secure_field_point point = sample_batches[tid].point;
            qm31 value = sampled_value; 

            size_t sampled_offset = sample_batches_offset + (j * 3);
            complex_conjugate_line_coeffs(point, value, alpha, &flattened_line_coeffs[sampled_offset], &flattened_line_coeffs[sampled_offset + 1], &flattened_line_coeffs[sampled_offset + 2]); 
        }
    }
}

// __device__ void denominator_inverse(point domain_point, column_sample_batch *sample_batches, cm31 *result) {
//     result[0] = {1234450342, 2089936180}; // Result of denominator_inverse(sample_batches, domain.at(0))
// }

__device__ void denominator_inverse(
        column_sample_batch *sample_batches,
        uint32_t sample_size,
        const point domain_point,
        cm31 *flat_denominators) {

    for (unsigned int i = 0; i < sample_size; i++) {
        cm31 prx = sample_batches[i].point.x.a;
        cm31 pry = sample_batches[i].point.y.a;
        cm31 pix = sample_batches[i].point.x.b;
        cm31 piy = sample_batches[i].point.y.b;

        cm31 first_substraction = {sub(prx.a, domain_point.x), prx.b};
        cm31 second_substraction = {sub(pry.a, domain_point.y), pry.b};
        cm31 result = sub(mul(first_substraction, piy),
                          mul(second_substraction, pix));
        flat_denominators[i] = inv(result);
    }
}

__global__ void accumulate_quotients_in_gpu(
        uint32_t half_coset_initial_index,
        uint32_t half_coset_step_size,
        uint32_t domain_size,
        int domain_log_size,
        m31 **columns,
        uint32_t number_of_columns,
        qm31 random_coefficient,
        column_sample_batch *sample_batches,
        uint32_t sample_size,
        uint32_t *result_column_0,
        uint32_t *result_column_1,
        uint32_t *result_column_2,
        uint32_t *result_column_3,
        qm31 *flattened_line_coeffs,
        uint32_t *line_coeffs_sizes,
        qm31 *batch_random_coeffs,
        cm31 *denominator_inverses
) {
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    denominator_inverses = &denominator_inverses[row * sample_size];

    if (row < domain_size) {
        uint32_t domain_index = bit_reverse(row, domain_log_size);
        point domain_point = domain_at_index(half_coset_initial_index, half_coset_step_size, domain_index, domain_size);

        denominator_inverse(
            sample_batches,
            sample_size,
            domain_point,
            denominator_inverses
        );

        int i = 0;

        qm31 row_accumulator = qm31{cm31{0, 0}, cm31{0, 0}};
        int line_coeffs_offset = 0;
        while (i < sample_size) {
            column_sample_batch sample_batch = sample_batches[i];
            qm31 *line_coeffs = &flattened_line_coeffs[line_coeffs_offset * 3];
            qm31 batch_coeff = batch_random_coeffs[i];
            int line_coeffs_size = line_coeffs_sizes[i];

            qm31 numerator = qm31{cm31{0, 0}, cm31{0, 0}};
            for(int j = 0; j < line_coeffs_size; j++) {
                qm31 a = line_coeffs[3 * j + 0];
                qm31 b = line_coeffs[3 * j + 1];
                qm31 c = line_coeffs[3 * j + 2];

                int column_index = sample_batches[i].columns[j];
                qm31 linear_term = add(mul_by_scalar(a, domain_point.y), b);
                qm31 value = mul_by_scalar(c, columns[column_index][row]);

                numerator = add(numerator, sub(value, linear_term));
            }

            row_accumulator = add(mul(row_accumulator, batch_coeff), mul(numerator, denominator_inverses[i]));
            line_coeffs_offset += line_coeffs_size;
            i++;
        }

        result_column_0[row] = row_accumulator.a.a;
        result_column_1[row] = row_accumulator.a.b;
        result_column_2[row] = row_accumulator.b.a;
        result_column_3[row] = row_accumulator.b.b;

    }
}

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
        uint32_t flattened_line_coeffs_size
) {
    int domain_log_size = log_2((int)domain_size);

    auto sample_batches = (column_sample_batch *)malloc(sizeof(column_sample_batch) * sample_size);

    column_sample_batch *sample_batches_device;
    cudaMalloc((void**)&sample_batches_device, sizeof(column_sample_batch) * sample_size);
    cm31* denominator_inverses;

    cudaMalloc((void**)&denominator_inverses, sizeof(cm31) * sample_size * domain_size);

    uint32_t *sample_column_indexes_device;
    cudaMalloc((void**)&sample_column_indexes_device, sizeof(uint32_t) * sample_column_indexes_size);
    cudaMemcpy(sample_column_indexes_device, sample_column_indexes, sizeof(uint32_t) * sample_column_indexes_size, cudaMemcpyHostToDevice);

    qm31 *sample_column_values_device;
    cudaMalloc((void**)&sample_column_values_device, sizeof(qm31) * sample_column_indexes_size);
    cudaMemcpy(sample_column_values_device, sample_column_values, sizeof(qm31) * sample_column_indexes_size, cudaMemcpyHostToDevice);

    column_sample_batches_for(
            sample_points,
            sample_column_indexes_device,
            sample_column_values_device,
            sample_column_and_values_sizes,
            sample_size,
            sample_batches
    );

    cudaMemcpy(sample_batches_device, sample_batches, sizeof(column_sample_batch) * sample_size, cudaMemcpyHostToDevice);

    qm31 *batch_random_coeffs_device;
    cudaMalloc((void**)&batch_random_coeffs_device, sizeof(qm31) * sample_size);

    uint32_t *line_coeffs_sizes_device;
    cudaMalloc((void**)&line_coeffs_sizes_device, sizeof(uint32_t) * sample_size);

    qm31 *flattened_line_coeffs_device;
    cudaMalloc((void**)&flattened_line_coeffs_device, sizeof(qm31) * flattened_line_coeffs_size);

    // Accumulate Quotient Constants
    int block_dim = sample_size < THREAD_MAX_COUNT ? sample_size : THREAD_MAX_COUNT; 
    int num_blocks = block_dim < THREAD_MAX_COUNT ? 1 : (sample_size + block_dim - 1) / block_dim;
    column_line_and_batch_random_coeffs<<<num_blocks, block_dim>>>(
            sample_batches_device, 
            sample_size, 
            random_coefficient,
            flattened_line_coeffs_device, 
            line_coeffs_sizes_device,
            batch_random_coeffs_device
    );

    // TODO: set to 1024
    block_dim = 512;
    num_blocks = (domain_size + block_dim - 1) / block_dim;
    accumulate_quotients_in_gpu<<<num_blocks, block_dim>>>(
            half_coset_initial_index,
            half_coset_step_size,
            domain_size,
            domain_log_size,
            columns,
            number_of_columns,
            random_coefficient,
            sample_batches_device,
            sample_size,
            result_column_0,
            result_column_1,
            result_column_2,
            result_column_3,
            flattened_line_coeffs_device,
            line_coeffs_sizes_device,
            batch_random_coeffs_device,
            denominator_inverses
    );
    cudaDeviceSynchronize();

    free(sample_batches);
    cudaFree(sample_batches_device);
    cudaFree(denominator_inverses);
    cudaFree(sample_column_indexes_device);
    cudaFree(sample_column_values_device);
    cudaFree(batch_random_coeffs_device);
    cudaFree(line_coeffs_sizes_device);
    cudaFree(flattened_line_coeffs_device);
}
