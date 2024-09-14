#include "example.cuh"
#include "utils.cuh"

#include <cstdio>

__device__ m31 eval_coset_vanishing(point step_sqrt, point eval_point, int log_size) {
    point p = point_mul(eval_point, step_sqrt);
    m31 x = p.x;
    int i = 1;
    while (i < log_size) {
        m31 x_sq = mul(x, x);
        x = sub(add(x_sq, x_sq), m31{1});
        i++;
    }
    return x;
}

__device__ point compute_point(int exponent, point initial_point, point step_point) {
    point p = point_pow(step_point, exponent);
    return point_mul(initial_point, p);
}

__host__ point compute_claim_value_point(int trace_length, int offset) {
    point g = point_of_order(trace_length);
    point g_inv = point_inv(g);
    point r = point_pow(g_inv, offset);
    return point_mul(g_inv, r);
}

__device__ int mod_nonnegative(int n, int mod) {
    int result = n % mod;
    if (result < 0) {
        result += mod;
    }
    return result;
}

__global__ void evaluate_constraints(
    m31 *evals,
    int evals_size,
    int log_trace_length,
    m31 *output_0,
    m31 *output_1,
    m31 *output_2,
    m31 *output_3,
    int output_size,
    int direction,
    int direction_offset,
    int chunk_offset,
    m31 claim_value,
    point initial_point,
    point coset_step_sqrt,
    point step_point,
    point cv_point_0,
    point cv_point_1,
    qm31 random_coeff_0,
    qm31 random_coeff_1
) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int output_index = bit_reverse(id + chunk_offset, log_trace_length + 1);
    int trace_length = evals_size / 2;

    if (id < evals_size) {
        point current_point = compute_point(id, initial_point, step_point);

        m31 e0 = evals[mod_nonnegative(direction_offset + evals_size - 1 - id, evals_size)];
        m31 e1 = evals[mod_nonnegative(direction_offset + evals_size - 1 - id - direction, evals_size)];
        m31 e2 = evals[mod_nonnegative(direction_offset + evals_size - 1 - id - 2 * direction, evals_size)];
        

        // Compute boundary constraint
        // 1 + y * (claim - 1) * p.y^-1
        m31 linear = add(m31{ 1 }, mul(current_point.y, mul(sub(claim_value, m31{ 1 } ), inv(cv_point_0.y))));

        m31 numerator_1 = sub(e0, linear);
        m31 denom_first_minor = mul(cv_point_0.y, current_point.x);
        m31 denom_second_minor = mul(sub(m31{1}, cv_point_0.x), current_point.y);
        m31 denom_third_minor = neg(cv_point_0.y);
        m31 denominator_1 = add(add(denom_first_minor, denom_second_minor), denom_third_minor);
        qm31 res_1 = mul(mul(numerator_1, inv(denominator_1)), random_coeff_0);

        // Compute transition constraint
        m31 constraint_value = sub(add(mul(e0, e0), mul(e1, e1)), e2);
        m31 selector_first_minor = mul(current_point.x, sub(cv_point_1.y, cv_point_0.y));
        m31 selector_second_minor = mul(current_point.y, sub(cv_point_0.x, cv_point_1.x));
        m31 selector_third_minor = sub(mul(cv_point_1.x, cv_point_0.y), mul(cv_point_0.x, cv_point_1.y));
        m31 selector = add(add(selector_first_minor, selector_second_minor), selector_third_minor);
        m31 numerator_2 = mul(constraint_value, selector);
        m31 denominator_2 = eval_coset_vanishing(coset_step_sqrt, current_point, log_trace_length);
        qm31 res_2 = mul(mul(numerator_2, inv(denominator_2)), random_coeff_1);
        qm31 res = add(res_1, res_2);

        // Accumulate
        output_0[output_index] = res.a.a;
        output_1[output_index] = res.a.b;
        output_2[output_index] = res.b.a;
        output_3[output_index] = res.b.b;
    }
}

void fibonacci_component_evaluate_constraint_quotients_on_domain(
    m31 *evals,
    int evals_size,
    m31 *output_0,
    m31 *output_1,
    m31 *output_2,
    m31 *output_3,
    m31 claim_value,
    point initial_point,
    point step_point,
    qm31 random_coeff_0,
    qm31 random_coeff_1
) {
    int trace_length = evals_size / 2;
    int log_trace_length = log_2(trace_length);
    int block_dim = 1024;
    int num_blocks = (evals_size / 2 + block_dim - 1) / block_dim;

    int output_size = evals_size;
    point claim_value_point_0 = compute_claim_value_point(trace_length, 0);
    point claim_value_point_1 = compute_claim_value_point(trace_length, 1);
    point coset_point_sqrt = point_mul(initial_point, initial_point);

    // Primer paso con evals[.. evals_size / 2]
    evaluate_constraints<<<num_blocks, block_dim>>>(
        &evals[evals_size / 2],
        evals_size / 2,
        log_trace_length,
        output_0,
        output_1,
        output_2,
        output_3,
        output_size,
        1,
        0,
        0,
        claim_value,
        initial_point,
        coset_point_sqrt,
        step_point,
        claim_value_point_0,
        claim_value_point_1,
        random_coeff_0,
        random_coeff_1
    );

    // Segundo paso con evals[evals_size / 2 ..]
    evaluate_constraints<<<num_blocks, block_dim>>>(
        evals,
        evals_size / 2,
        log_trace_length,
        output_0,
        output_1,
        output_2,
        output_3,
        output_size,
        -1,
        1,
        evals_size / 2 ,
        claim_value,
        point_inv(initial_point),
        coset_point_sqrt,
        point_inv(step_point),
        claim_value_point_0,
        claim_value_point_1,
        random_coeff_0,
        random_coeff_1
    );

    cudaDeviceSynchronize();
}