#include "../include/circle.cuh"
#include "../include/bit_reverse.cuh"
#include "../include/utils.cuh"



__device__ int get_twiddle(m31 *twiddles, int index) {
    int k = index >> 2;
    if (index % 4 == 0) {
        return twiddles[2 * k + 1];
    } else if (index % 4 == 1) {
        return neg(twiddles[2 * k + 1]);
    } else if (index % 4 == 2) {
        return neg(twiddles[2 * k]);
    } else {
        return twiddles[2 * k];
    }
}

__global__ void ifft_circle_part(m31 *values, m31 *inverse_twiddles_tree, int values_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (values_size >> 1)) {
        m31 val0 = values[2 * idx];
        m31 val1 = values[2 * idx + 1];
        m31 twiddle = get_twiddle(inverse_twiddles_tree, idx);

        values[2 * idx] = add(val0, val1);
        values[2 * idx + 1] = mul(sub(val0, val1), twiddle);
    }
}

__global__ void ifft_line_part(m31 *values, m31 *inverse_twiddles_tree, int values_size, int inverse_twiddles_size,
                               int layer_domain_offset, int layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (values_size >> 1)) {
        int number_polynomials = 1 << layer;
        int h = idx >> layer;
        int l = idx & (number_polynomials - 1);
        int idx0 = (h << (layer + 1)) + l;
        int idx1 = idx0 + number_polynomials;

        m31 val0 = values[idx0];
        m31 val1 = values[idx1];
        m31 twiddle = inverse_twiddles_tree[layer_domain_offset + h];

        values[idx0] = add(val0, val1);
        values[idx1] = mul(sub(val0, val1), twiddle);
    }
}

__global__ void rfft_circle_part(m31 *values, m31 *inverse_twiddles_tree, int values_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;


    if (idx < (values_size >> 1)) {
        m31 val0 = values[2 * idx];
        m31 val1 = values[2 * idx + 1];
        m31 twiddle = get_twiddle(inverse_twiddles_tree, idx);

        m31 temp = mul(val1, twiddle);

        values[2 * idx] = add(val0, temp);
        values[2 * idx + 1] = sub(val0, temp);
    }
}

__global__ void rfft_line_part(m31 *values, m31 *inverse_twiddles_tree, int values_size, int inverse_twiddles_size,
                               int layer_domain_offset, int layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (values_size >> 1)) {
        int number_polynomials = 1 << layer;
        int h = idx / number_polynomials;
        int l = idx % number_polynomials;
        int idx0 = (h << (layer + 1)) + l;
        int idx1 = idx0 + number_polynomials;

        m31 val0 = values[idx0];
        m31 val1 = values[idx1];
        m31 twiddle = inverse_twiddles_tree[layer_domain_offset + h];

        m31 temp = mul(val1, twiddle);

        values[idx0] = add(val0, temp);
        values[idx1] = sub(val0, temp);
    }
}

__global__ void rescale(m31 *values, int size, m31 factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        values[idx] = mul(values[idx], factor);
    }
}

void
interpolate(int eval_domain_size, m31 *values, m31 *inverse_twiddles_tree, int inverse_twiddles_size, int values_size) {
    inverse_twiddles_tree = &inverse_twiddles_tree[inverse_twiddles_size - eval_domain_size];
    int block_dim = 256;
    int num_blocks = ((values_size >> 1) + block_dim - 1) / block_dim;
    ifft_circle_part<<<num_blocks, block_dim>>>(values, inverse_twiddles_tree, values_size);

    int log_values_size = log_2(values_size);
    int layer_domain_size = values_size >> 1;
    int layer_domain_offset = 0;
    int i = 1;
    while (i < log_values_size) {
        ifft_line_part<<<num_blocks, block_dim>>>(values, inverse_twiddles_tree, values_size, layer_domain_size, layer_domain_offset, i);

        layer_domain_size >>= 1;
        layer_domain_offset += layer_domain_size;
        i += 1;
    }
    cudaDeviceSynchronize();

    block_dim = 1024;
    num_blocks = (values_size + block_dim - 1) / block_dim;
    m31 factor = inv(pow(m31{2}, log_values_size));
    rescale<<<num_blocks, block_dim>>>(values, values_size, factor);
    cudaDeviceSynchronize();
}

__global__ void batch_ifft_circle_part(m31 **values, m31 *inverse_twiddles_tree, int values_size, int number_of_rows) {
    int index = blockIdx.y * blockDim.x + threadIdx.x;
    unsigned int column_index = blockIdx.x;

    if (index < (number_of_rows >> 1) && column_index < values_size) {
        m31 *column = values[column_index];

        m31 val0 = column[2 * index];
        m31 val1 = column[2 * index + 1];
        m31 twiddle = get_twiddle(inverse_twiddles_tree, index);

        column[2 * index] = add(val0, val1);
        column[2 * index + 1] = mul(sub(val0, val1), twiddle);
    }
}

__global__ void
batch_ifft_line_part(m31 **values, m31 *twiddles, int values_size, int number_of_rows, int layer) {
    int index = blockIdx.y * blockDim.x + threadIdx.x;
    unsigned int column_index = blockIdx.x;

    if (index < (number_of_rows >> 1) && column_index < values_size) {
        // `index` is in [0, number_of_rows / 2).
        // It is interpreted as the n - 1 bit-string `twiddle_index || polynomial_index`,
        // where n = log_2(`number_of_rows`), `polynomial_index` is the rightmost `layer` bits,
        // and `twiddle_index` is the rest `n - layer - 1` bits.
        // This thread performs a butterfly between the values at indexes `twiddle_index || 0 || polynomial_index`
        // and `twiddle_index || 1 || polynomial_index`.
        m31 *column = values[column_index];

        int number_polynomials = 1 << layer;
        int twiddle_index = index >> layer;
        int polynomial_index = index & (number_polynomials - 1);

        int idx0 = (twiddle_index << (layer + 1)) | polynomial_index;
        int idx1 = idx0 | number_polynomials;

        m31 val0 = column[idx0];
        m31 val1 = column[idx1];

        m31 twiddle = twiddles[twiddle_index];

        column[idx0] = add(val0, val1);
        column[idx1] = mul(sub(val0, val1), twiddle);
    }
}

__global__ void batch_rescale(m31 **values, int values_size, int number_of_rows, m31 factor) {
    int index = blockIdx.y * blockDim.x + threadIdx.x;
    unsigned int column_index = blockIdx.x;

    if (index < number_of_rows && column_index < values_size) {
        values[column_index][index] = mul(values[column_index][index], factor);
    }
}

void interpolate_columns(int eval_domain_size, m31 **values, m31 *inverse_twiddles_tree, int inverse_twiddles_size,
                         int values_size, int number_of_rows) {
    // TODO: Handle case where columns are of different sizes.
    int blockDimensions = 1024;

    m31 **device_values = clone_to_device<m31*>(values, values_size);

    m31 *inverseTwiddlesTree = inverse_twiddles_tree;
    inverseTwiddlesTree = &inverseTwiddlesTree[inverse_twiddles_size - eval_domain_size];
    int numBlocks = ((number_of_rows >> 1) + blockDimensions - 1) / blockDimensions;
    dim3 gridDimensions(values_size, numBlocks);

    batch_ifft_circle_part<<<gridDimensions, blockDimensions>>>(device_values, inverseTwiddlesTree, values_size, number_of_rows);
    cudaDeviceSynchronize();

    int log_number_of_rows = log_2(number_of_rows);
    int layer_domain_size = number_of_rows >> 1;
    int layer_domain_offset = 0;
    int i = 1;
    while (i < log_number_of_rows) {
        batch_ifft_line_part<<<gridDimensions, blockDimensions>>>(device_values, &inverseTwiddlesTree[layer_domain_offset], values_size, number_of_rows, i);
        layer_domain_size >>= 1;
        layer_domain_offset += layer_domain_size;
        i += 1;
    }
    cudaDeviceSynchronize();

    m31 factor = inv(pow(m31{2}, log_number_of_rows));
    numBlocks = (number_of_rows + blockDimensions - 1) / blockDimensions;
    dim3 rescaleGridDimensions(values_size, numBlocks);
    batch_rescale<<<rescaleGridDimensions, blockDimensions>>>(device_values, values_size, number_of_rows, factor);
    cudaDeviceSynchronize();
    
    cuda_free_memory(device_values);
}

void evaluate(int eval_domain_size, m31 *values, m31 *twiddles_tree, int twiddles_size, int values_size) {
    twiddles_tree = &twiddles_tree[twiddles_size - eval_domain_size];
    int block_dim = 256;
    int num_blocks = ((values_size >> 1) + block_dim - 1) / block_dim;

    int log_values_size = log_2(values_size);
    int layer_domain_size = 1;
    int layer_domain_offset = (values_size >> 1) - 2;
    int i = log_values_size - 1;
    while (i > 0) {
        rfft_line_part<<<num_blocks, block_dim>>>(values, twiddles_tree, values_size, layer_domain_size,
                                                  layer_domain_offset, i);
        layer_domain_size <<= 1;
        layer_domain_offset -= layer_domain_size;
        i -= 1;
    }

    rfft_circle_part<<<num_blocks, block_dim>>>(values, twiddles_tree, values_size);
    cudaDeviceSynchronize();
}

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

__global__ void
eval_at_point_second_pass(qm31 *temp, qm31 *factors, int level_size, int factor_offset, int level_offset,
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

