#include "../include/utils.cuh"

__global__ void gen_eq_evals_kernel(qm31 v, qm31 *y, uint32_t y_size, qm31 *evals, uint32_t evals_size) {
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    evals[thread_index] = v;
}

extern "C"
void gen_eq_evals(qm31 v, qm31 *y, uint32_t y_size, qm31 *evals, uint32_t evals_size) {
    const unsigned int BLOCK_SIZE = 1024;
    const unsigned int NUMBER_OF_BLOCKS = (evals_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    gen_eq_evals_kernel<<<NUMBER_OF_BLOCKS, min(evals_size, BLOCK_SIZE)>>>(v, y, y_size, evals, evals_size);
}
