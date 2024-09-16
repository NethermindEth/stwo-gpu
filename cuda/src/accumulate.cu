#include "accumulate.cuh"
#include "utils.cuh"

__global__
void accumulate_kernel(int size, m31 **left_columns, m31 **right_columns) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        left_columns[0][i] = add(left_columns[0][i], right_columns[0][i]);
        left_columns[1][i] = add(left_columns[1][i], right_columns[1][i]);
        left_columns[2][i] = add(left_columns[2][i], right_columns[2][i]);
        left_columns[3][i] = add(left_columns[3][i], right_columns[3][i]);
    }
}

void accumulate(int size, m31 **left_columns, m31 **right_columns) {
    m31 **left_columns_device = clone_to_device<m31*>(left_columns, 4);
    m31 **right_columns_device = clone_to_device<m31*>(right_columns, 4);

    int block_dim = 1024;
    int num_blocks = (size + block_dim - 1) / block_dim;
    accumulate_kernel<<<num_blocks, block_dim>>>(size, left_columns_device, right_columns_device);
    cudaDeviceSynchronize();

    cuda_free_memory(left_columns_device);
    cuda_free_memory(right_columns_device);
}
