#include "poly/rfft.cuh"
#include "poly/utils.cuh"
#include "utils.cuh"

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

__global__ void batch_rfft_circle_part(m31 **values, m31 *inverse_twiddles_tree, int number_of_columns, int number_of_rows) {
    int idx = blockIdx.y * blockDim.x + threadIdx.x;
    unsigned int column_index = blockIdx.x;

    if (idx < (number_of_rows >> 1) && column_index < number_of_columns) {
        m31 *column = values[column_index];

        m31 val0 = column[2 * idx];
        m31 val1 = column[2 * idx + 1];
        m31 twiddle = get_twiddle(inverse_twiddles_tree, idx);

        m31 temp = mul(val1, twiddle);

        column[2 * idx] = add(val0, temp);
        column[2 * idx + 1] = sub(val0, temp);
    }
}

__global__ void batch_rfft_line_part(
        m31 **values, m31 *inverse_twiddles_tree, int number_of_columns, int number_of_rows, int layer_domain_offset, int layer
) {
    int idx = blockIdx.y * blockDim.x + threadIdx.x;
    unsigned int column_index = blockIdx.x;

    if (idx < (number_of_rows >> 1) && column_index < number_of_columns) {
        m31 *column = values[column_index];

        int number_polynomials = 1 << layer;
        int h = idx / number_polynomials;
        int l = idx % number_polynomials;
        int idx0 = (h << (layer + 1)) + l;
        int idx1 = idx0 + number_polynomials;

        m31 val0 = column[idx0];
        m31 val1 = column[idx1];
        m31 twiddle = inverse_twiddles_tree[layer_domain_offset + h];

        m31 temp = mul(val1, twiddle);

        column[idx0] = add(val0, temp);
        column[idx1] = sub(val0, temp);
    }
}

void evaluate_columns(const int *eval_domain_sizes, m31 **values, m31 *twiddles_tree, int twiddles_size, int number_of_columns, const int *column_sizes) {
    // TODO: Handle case where columns are of different sizes.
    int number_of_rows = column_sizes[0];
    int eval_domain_size = eval_domain_sizes[0];

    m31 **device_values = clone_to_device<m31*>(values, number_of_columns);

    twiddles_tree = &twiddles_tree[twiddles_size - eval_domain_size];

    int block_size = 1024;
    int number_of_blocks = ((number_of_rows >> 1) + block_size - 1) / block_size;
    dim3 grid_dimensions(number_of_columns, number_of_blocks);

    int log_number_of_rows = log_2(number_of_rows);
    int layer_domain_size = 1;
    int layer_domain_offset = (number_of_rows >> 1) - 2;
    int i = log_number_of_rows - 1;

    while (i > 0) {
        batch_rfft_line_part<<<grid_dimensions, block_size>>>(
                device_values, twiddles_tree, number_of_columns, number_of_rows, layer_domain_offset, i
        );
        layer_domain_size <<= 1;
        layer_domain_offset -= layer_domain_size;
        i -= 1;
    }

    batch_rfft_circle_part<<<grid_dimensions, block_size>>>(device_values, twiddles_tree, number_of_columns, number_of_rows);
    cudaDeviceSynchronize();

    cuda_free_memory(device_values);
}
