#include "../include/utils.cuh"
#include <cstdio>  // TODO: Remove

__global__ void gen_eq_evals_kernel(qm31 v, qm31 *y, uint32_t y_size, qm31 *evals) {
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    qm31 qm31_identity = qm31{cm31{m31{1}, m31{0}}, cm31{m31{0}, m31{0}}};

    qm31 eq_eval = v;
    for (unsigned int x_coordinate_index = 0; x_coordinate_index < y_size; x_coordinate_index++) {
        m31 x_coordinate_value = m31{thread_index >> x_coordinate_index & 1};
        qm31 &y_value = y[x_coordinate_index];
        qm31 left_summand = mul_by_scalar(y_value, x_coordinate_value);
        qm31 right_summand = mul_by_scalar(sub(qm31_identity, y_value), m31{1} - x_coordinate_value);
        eq_eval = mul(eq_eval, add(left_summand, right_summand));
    }

    evals[thread_index] = eq_eval;
}

extern "C"
void gen_eq_evals(qm31 v, qm31 *y, uint32_t y_size, qm31 *evals, uint32_t evals_size) {
    const unsigned int BLOCK_SIZE = 1024;
    const unsigned int NUMBER_OF_BLOCKS = (evals_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    gen_eq_evals_kernel<<<NUMBER_OF_BLOCKS, min(evals_size, BLOCK_SIZE)>>>(v, y, y_size, evals);
}
