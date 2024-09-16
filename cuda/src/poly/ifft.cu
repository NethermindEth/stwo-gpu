#include "poly/ifft.cuh"

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

__global__ void ifft_line_part(m31 *values, m31 *twiddles, int values_size, int layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (values_size >> 1)) {
        // `index` is in [0, values_size / 2).
        // It is interpreted as the n - 1 bit-string `twiddle_index || polynomial_index`,
        // where n = log_2(`values_size`), `polynomial_index` is the rightmost `layer` bits,
        // and `twiddle_index` is the rest `n - layer - 1` bits.
        // This thread performs a butterfly between the values at indexes `twiddle_index || 0 || polynomial_index`
        // and `twiddle_index || 1 || polynomial_index`.
        int number_polynomials = 1 << layer;
        int twiddle_index = idx >> layer;
        int l = idx & (number_polynomials - 1);
        int idx0 = (twiddle_index << (layer + 1)) + l;
        int idx1 = idx0 + number_polynomials;

        m31 val0 = values[idx0];
        m31 val1 = values[idx1];
        m31 twiddle = twiddles[twiddle_index];

        values[idx0] = add(val0, val1);
        values[idx1] = mul(sub(val0, val1), twiddle);
    }
}

void interpolate(int eval_domain_size, m31 *values, m31 *inverse_twiddles_tree, int inverse_twiddles_size, int values_size) {
    inverse_twiddles_tree = &inverse_twiddles_tree[inverse_twiddles_size - eval_domain_size];
    int block_dim = 256;
    int num_blocks = ((values_size >> 1) + block_dim - 1) / block_dim;
    ifft_circle_part<<<num_blocks, block_dim>>>(values, inverse_twiddles_tree, values_size);

    int log_values_size = log_2(values_size);
    int layer_domain_size = values_size >> 1;
    int layer_domain_offset = 0;
    int i = 1;
    while (i < log_values_size) {
        ifft_line_part<<<num_blocks, block_dim>>>(values, &inverse_twiddles_tree[layer_domain_offset], values_size, i);

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

__global__
void batch_ifft_line_part(m31 **values, m31 *twiddles, int values_size, int number_of_rows, int layer) {
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
        batch_ifft_line_part<<<gridDimensions, blockDimensions>>>(
            device_values,
            &inverseTwiddlesTree[layer_domain_offset],
            values_size,
            number_of_rows,
            i
        );
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