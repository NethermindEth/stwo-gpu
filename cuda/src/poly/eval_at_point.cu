#include "poly/eval_at_point.cuh"
#include "utils.cuh"


__global__ void eval_at_point_first_pass(m31 *g_coeffs, qm31 *temp, qm31 *factors, int coeffs_size, int factors_size,
                                         int output_offset) {
    int idx = threadIdx.x;

    qm31 *output = &temp[output_offset];

    // Thread syncing happens within a block. 
    // Split the problem to feed them to multiple blocks.
    if (coeffs_size >= 512) {
        coeffs_size = 512;
    }

    extern __shared__ m31 s_coeffs[];
    extern __shared__ qm31 s_level[];

    s_coeffs[idx] = g_coeffs[2 * blockIdx.x * blockDim.x + idx];
    s_coeffs[idx + blockDim.x] = g_coeffs[2 * blockIdx.x * blockDim.x + idx + blockDim.x];
    __syncthreads();

    int level_size = coeffs_size >> 1;
    int factor_idx = factors_size - 1;

    if (idx < level_size) {
        m31 alpha = s_coeffs[2 * idx];
        m31 beta = s_coeffs[2 * idx + 1];
        qm31 factor = factors[factor_idx];

        qm31 result = {
                {add(mul(beta, factor.a.a), alpha), mul(factor.a.b, beta)},
                {mul(beta, factor.b.a),             mul(beta, factor.b.b)}
        };
        s_level[idx] = result;
    }
    factor_idx -= 1;
    level_size >>= 1;

    while (level_size > 0) {
        if (idx < level_size) {
            __syncthreads();
            qm31 a = s_level[2 * idx];
            qm31 b = s_level[2 * idx + 1];
            __syncthreads();
            s_level[idx] = add(a, mul(b, factors[factor_idx]));
        }
        factor_idx -= 1;
        level_size >>= 1;

    }

    if (idx == 0) {
        output[blockIdx.x] = s_level[0];
    }
}

__global__
void eval_at_point_second_pass(qm31 *temp, qm31 *factors, int level_size, int factor_offset, int level_offset,
                          int output_offset) {
    int idx = threadIdx.x;

    qm31 *level = &temp[level_offset];
    qm31 *output = &temp[output_offset];

    // Thread syncing happens within a block. 
    // Split the problem to feed them to multiple blocks.
    if (level_size >= 512) {
        level_size = 512;
    }

    extern __shared__ qm31 s_level[];

    s_level[idx] = level[2 * blockIdx.x * blockDim.x + idx];
    s_level[idx + blockDim.x] = level[2 * blockIdx.x * blockDim.x + idx + blockDim.x];

    level_size >>= 1;

    int factor_idx = factor_offset;

    while (level_size > 0) {
        if (idx < level_size) {
            __syncthreads();
            qm31 a = s_level[2 * idx];
            qm31 b = s_level[2 * idx + 1];
            __syncthreads();
            s_level[idx] = add(a, mul(b, factors[factor_idx]));
        }
        factor_idx -= 1;
        level_size >>= 1;
    }

    if (idx == 0) {
        output[blockIdx.x] = s_level[0];
    }
}

qm31 eval_at_point(m31 *coeffs, int coeffs_size, qm31 point_x, qm31 point_y) {
    int log_coeffs_size = log_2(coeffs_size);

    qm31 *host_mappings = (qm31 *) malloc(sizeof(qm31) * log_coeffs_size);
    host_mappings[log_coeffs_size - 1] = point_y;
    host_mappings[log_coeffs_size - 2] = point_x;
    qm31 x = point_x;
    for (int i = 2; i < log_coeffs_size; i += 1) {
        x = sub(mul(qm31{cm31{2, 0}, cm31{0, 0}}, mul(x, x)), qm31{cm31{1, 0}, cm31{0, 0}});
        host_mappings[log_coeffs_size - 1 - i] = x;
    }

    int temp_memory_size = 0;
    int size = coeffs_size;
    while (size > 1) {
        size = (size + 511) / 512;
        temp_memory_size += size;
    }

    qm31 *temp = cuda_malloc<qm31>(temp_memory_size);
    qm31 *device_mappings = clone_to_device<qm31>(host_mappings, log_coeffs_size);

    free(host_mappings);

    // First pass
    int block_dim = 256;
    int num_blocks = ((coeffs_size >> 1) + block_dim - 1) / block_dim;
    int shared_memory_bytes = 512 * 4 + 512 * 8;
    int output_offset = temp_memory_size - num_blocks;

    eval_at_point_first_pass<<<num_blocks, block_dim, shared_memory_bytes>>>(coeffs, temp, device_mappings, coeffs_size,
                                                                             log_coeffs_size, output_offset);

    // Second pass
    int mappings_offset = log_coeffs_size - 1;
    int level_offset = output_offset;
    while (num_blocks > 1) {
        mappings_offset -= 9;
        int new_num_blocks = ((num_blocks >> 1) + block_dim - 1) / block_dim;
        shared_memory_bytes = 512 * 4 * 4;
        output_offset = level_offset - new_num_blocks;
        eval_at_point_second_pass<<<new_num_blocks, block_dim, shared_memory_bytes>>>(temp, device_mappings, num_blocks,
                                                                                      mappings_offset, level_offset,
                                                                                      output_offset);
        num_blocks = new_num_blocks;
        level_offset = output_offset;
    }

    qm31 result = qm31{cm31{0, 0}, cm31{0, 1}};
    cudaDeviceSynchronize();

    cuda_mem_copy_device_to_host<qm31>(temp, &result, 1);

    cuda_free_memory(temp);
    cuda_free_memory(device_mappings);
    return result;
}

void evaluate_polynomials_out_of_domain(
    qm31 **result, m31 **polynomials, int *polynomial_sizes, int number_of_polynomials,
    qm31 **out_of_domain_points_x, qm31 **out_of_domain_points_y, int *sample_sizes
) {
    // column
    //     .into_iter()
    //     .enumerate()
    //     .map( |(index2, polynomial)|     // Index2 es el del thread!
    //         out_of_domain_points[index2]
    //             .iter()
    //             .map(|&point| PointSample {
    //                 point,
    //                 value: polynomial.eval_at_point(point),
    //             })
    //             .collect_vec()
    //     ).collect();

    // qm31 **device_result = cuda_malloc<qm31*>(number_of_polynomials);
    // m31 **device_polynomials = clone_to_device<m31*>(polynomials, number_of_polynomials);

    // int number_of_blocks = 0;
    // int block_size = 1024;
    // int shared_memory_bytes = 0;
    // eval_polys_at_points<<<number_of_blocks, block_size, shared_memory_bytes>>>(
    //     device_result, device_polynomials, polynomial_sizes, number_of_polynomials,
    //     out_of_domain_points_x, out_of_domain_points_y, sample_sizes
    // );

    // cuda_free_memory(device_result);
    // cuda_free_memory(device_polynomials);

    for (int index = 0; index < number_of_polynomials; index++) {
        m31 *polynomial = polynomials[index];
        int polynomial_size = polynomial_sizes[index];
        qm31 *poly_points_x = out_of_domain_points_x[index];
        qm31 *poly_points_y = out_of_domain_points_y[index];
        int sample_size = sample_sizes[index];

        for (int point_index = 0; point_index < sample_size; point_index++) {
            qm31 point_x = poly_points_x[point_index];
            qm31 point_y = poly_points_y[point_index];
            result[index][point_index] = eval_at_point(polynomial, polynomial_size, point_x, point_y);
        }
    }
};
