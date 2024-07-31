#include "../include/accumulate.cuh"

__global__ void
accumulate_aux(uint32_t *left_column_0, uint32_t *left_column_1, uint32_t *left_column_2, uint32_t *left_column_3,
               uint32_t *right_column_0, uint32_t *right_column_1, uint32_t *right_column_2,
               uint32_t *right_column_3);

void accumulate(uint32_t size,
                uint32_t *left_columns[],
                uint32_t *right_columns[]) {
    int block_dim = 1024;
    int num_blocks = (size + block_dim - 1) / block_dim;
    accumulate_aux<<<num_blocks, min(size, block_dim)>>>(
            left_columns[0], left_columns[1], left_columns[2], left_columns[3],
            right_columns[0], right_columns[1], right_columns[2], right_columns[3]);
    cudaDeviceSynchronize();
}

__global__ void
accumulate_aux(uint32_t *left_column_0, uint32_t *left_column_1, uint32_t *left_column_2, uint32_t *left_column_3,
               uint32_t *right_column_0, uint32_t *right_column_1, uint32_t *right_column_2,
               uint32_t *right_column_3) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    left_column_0[i] = add(left_column_0[i], right_column_0[i]);
    left_column_1[i] = add(left_column_1[i], right_column_1[i]);
    left_column_2[i] = add(left_column_2[i], right_column_2[i]);
    left_column_3[i] = add(left_column_3[i], right_column_3[i]);
}
