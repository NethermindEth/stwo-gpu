#include "../include/mle.cuh"
#include <stdio.h>
__global__ void fix_first_variable_kernel(m31 *evals, int evals_size, qm31 assignment, qm31* output_evals) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < (evals_size >> 1)) {
        m31 lhs_evals = evals[index];
        m31 rhs_evals = evals[index + (evals_size >> 1)];
        output_evals[index] = add(lhs_evals, mul(sub(rhs_evals, lhs_evals), assignment)); 
    }
}

void fix_first_variable(m31 *evals, int evals_size, qm31 assignment, qm31* output_evals) {
    int block_size = 1024;
    int num_blocks = ((evals_size >> 1) + block_size - 1) / block_size;
    fix_first_variable_kernel<<<num_blocks, block_size>>>(evals, evals_size, assignment, output_evals);
    cudaDeviceSynchronize();
}
