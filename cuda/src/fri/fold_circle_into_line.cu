#include "fri/fold_circle_into_line.cuh"
#include "fri/utils.cuh"
#include "poly/utils.cuh"


__global__ void fold_circle_into_line_kernel(
    m31 *domain,
    const uint32_t twiddle_offset,
    const uint32_t n,
    const qm31 alpha,
    const qm31 alpha_sq,
    m31 *eval_values_0,
    m31 *eval_values_1,
    m31 *eval_values_2,
    m31 *eval_values_3,
    m31 *folded_values_0,
    m31 *folded_values_1,
    m31 *folded_values_2,
    m31 *folded_values_3
) {
    const uint32_t *eval_values[4] = {eval_values_0,
                                      eval_values_1,
                                      eval_values_2,
                                      eval_values_3};

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    domain = &domain[twiddle_offset];

    if (i < (n >> 1)) {
        const uint32_t x_inverse = get_twiddle(domain, i);

        const uint32_t index_left = 2 * i;
        const uint32_t index_right = index_left + 1;

        const qm31 f_x = getEvaluation(eval_values, index_left);
        const qm31 f_x_minus = getEvaluation(eval_values, index_right);

        const qm31 f_0 = add(f_x, f_x_minus);
        const qm31 f_1 = mul_by_scalar(sub(f_x, f_x_minus), x_inverse);

        const qm31 f_prime = add(f_0, mul(alpha, f_1));

        qm31 previous_value = qm31 {
            folded_values_0[i],
            folded_values_1[i],
            folded_values_2[i],
            folded_values_3[i],
        };
        qm31 new_value = add(mul(previous_value, alpha_sq), f_prime);

        folded_values_0[i] = new_value.a.a;
        folded_values_1[i] = new_value.a.b;
        folded_values_2[i] = new_value.b.a;
        folded_values_3[i] = new_value.b.b;
    }
}

void fold_circle_into_line(m31 *gpu_domain, uint32_t twiddle_offset, uint32_t n, m31 *eval_values[], qm31 alpha, m31 *folded_values[]) {
    int block_dim = 1024;
    int num_blocks = (n / 2 + block_dim - 1) / block_dim;
    qm31 alpha_sq = mul(alpha, alpha);

    fold_circle_into_line_kernel<<<num_blocks, block_dim>>>(
            gpu_domain,
            twiddle_offset,
            n,
            alpha,
            alpha_sq,
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