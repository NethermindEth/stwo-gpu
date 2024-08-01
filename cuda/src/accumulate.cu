#include "../include/accumulate.cuh"

__global__ void
accumulate_kernel(uint32_t *left_columns[4],
                  uint32_t *right_columns[4]) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    left_columns[0][i] = add(left_columns[0][i], right_columns[0][i]);
    left_columns[1][i] = add(left_columns[1][i], right_columns[1][i]);
    left_columns[2][i] = add(left_columns[2][i], right_columns[2][i]);
    left_columns[3][i] = add(left_columns[3][i], right_columns[3][i]);
}

void accumulate(uint32_t size,
                uint32_t *left_columns[],
                uint32_t *right_columns[]) {
    int block_dim = 1024;
    int num_blocks = (size + block_dim - 1) / block_dim;
    accumulate_kernel<<<num_blocks, min(size, block_dim)>>>(left_columns, right_columns);
    cudaDeviceSynchronize();
}
