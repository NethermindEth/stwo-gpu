#include "../include/mle.cuh"
#include <stdio.h>

template<typename T>
__global__ void fix_first_variable_kernel(T *evals, int evals_size, qm31 assignment, qm31* output_evals) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < (evals_size >> 1)) {
        T lhs_evals = evals[index];
        T rhs_evals = evals[index + (evals_size >> 1)];
        output_evals[index] = add(lhs_evals, mul(sub(rhs_evals, lhs_evals), assignment)); 
    }
}

void fix_first_variable_basefield(m31 *evals, int evals_size, qm31 assignment, qm31* output_evals) {
    int block_size = 1024;
    int num_blocks = ((evals_size >> 1) + block_size - 1) / block_size;
    fix_first_variable_kernel<<<num_blocks, block_size>>>(evals, evals_size, assignment, output_evals);
    cudaDeviceSynchronize();
}

void fix_first_variable_securefield(qm31 *evals, int evals_size, qm31 assignment, qm31* output_evals) {
    int block_size = 1024;
    int num_blocks = ((evals_size >> 1) + block_size - 1) / block_size;
    fix_first_variable_kernel<<<num_blocks, block_size>>>(evals, evals_size, assignment, output_evals);
    cudaDeviceSynchronize();
}
