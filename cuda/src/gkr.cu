#include "../include/utils.cuh"

__global__ void gen_eq_evals_kernel(qm31 v, qm31 *y, uint32_t y_size, qm31 *evals) {
    // TODO: See if shared memory speeds this up

    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    qm31 eq_eval = v;
    unsigned int shifted_thread_index = thread_index;
    for (int i = y_size - 1; i >= 0; i--) {
        int offset = y_size * (shifted_thread_index & 1);
        eq_eval = mul(eq_eval, y[i + offset]);
        shifted_thread_index >>= 1;
    }
    evals[thread_index] = eq_eval;
}

extern "C"
void gen_eq_evals(qm31 v, qm31 *y, uint32_t y_size, qm31 *evals, uint32_t evals_size) {
    const unsigned int BLOCK_SIZE = 1024;
    const unsigned int NUMBER_OF_BLOCKS = (evals_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int precomputed_table_byte_length = sizeof(qm31) * y_size * 2;
    qm31 *precomputed_table = (qm31*)malloc(precomputed_table_byte_length);
    for(int i = 0; i < y_size; i++) {
        precomputed_table[i] = sub(m31{1}, y[i]);
        precomputed_table[i + y_size] = y[i];
    }

    qm31 *precomputed_table_device;
    cudaMalloc((void**)&precomputed_table_device, precomputed_table_byte_length);
    cudaMemcpy(precomputed_table_device, precomputed_table, precomputed_table_byte_length, cudaMemcpyHostToDevice);
    free(precomputed_table);

    gen_eq_evals_kernel<<<NUMBER_OF_BLOCKS, min(evals_size, BLOCK_SIZE)>>>(v, precomputed_table_device, y_size, evals);
    cudaDeviceSynchronize();
    cudaFree(precomputed_table_device);
}
