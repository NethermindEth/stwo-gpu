#include "fri/fold_line.cuh"
#include "fri/utils.cuh"

__device__ uint32_t f(const uint32_t *domain,
                      const uint32_t twiddle_offset,
                      const uint32_t i) {
    return domain[i + twiddle_offset];
}

__global__ void fold_applying(const uint32_t *domain,
                              const uint32_t twiddle_offset,
                              const uint32_t n,
                              const qm31 alpha,
                              uint32_t *eval_values_0,
                              uint32_t *eval_values_1,
                              uint32_t *eval_values_2,
                              uint32_t *eval_values_3,
                              uint32_t *folded_values_0,
                              uint32_t *folded_values_1,
                              uint32_t *folded_values_2,
                              uint32_t *folded_values_3) {
    const uint32_t *eval_values[4] = {eval_values_0,
                                      eval_values_1,
                                      eval_values_2,
                                      eval_values_3};

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n / 2) {
        const uint32_t x_inverse = f(domain, twiddle_offset, i);

        const uint32_t index_left = 2 * i;
        const uint32_t index_right = index_left + 1;

        const qm31 f_x = getEvaluation(eval_values, index_left);
        const qm31 f_x_minus = getEvaluation(eval_values, index_right);

        const qm31 f_0 = add(f_x, f_x_minus);
        const qm31 f_1 = mul_by_scalar(sub(f_x, f_x_minus), x_inverse);

        const qm31 f_prime = add(f_0, mul(alpha, f_1));

        folded_values_0[i] = f_prime.a.a;
        folded_values_1[i] = f_prime.a.b;
        folded_values_2[i] = f_prime.b.a;
        folded_values_3[i] = f_prime.b.b;
    }
}

void fold_line(uint32_t *gpu_domain,
               uint32_t twiddle_offset,
               uint32_t n,
               uint32_t **eval_values,
               qm31 alpha,
               uint32_t **folded_values) {
    int block_dim = 1024;
    int num_blocks = (n / 2 + block_dim - 1) / block_dim;
    fold_applying<<<num_blocks, block_dim>>>(
            gpu_domain,
            twiddle_offset,
            n,
            alpha,
            eval_values[0],
            eval_values[1],
            eval_values[2],
            eval_values[3],
            folded_values[0],
            folded_values[1],
            folded_values[2],
            folded_values[3]);
    cudaDeviceSynchronize();
}

