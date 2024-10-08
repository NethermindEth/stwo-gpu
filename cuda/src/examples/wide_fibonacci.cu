#include "../../include/examples/wide_fibonacci.cuh"

__global__ void evaluate_wide_fibonacci_constraint_quotients_kernel(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 **trace_evaluations,
    qm31 *numerators,
    qm31 *random_coeff_powers,
    m31 *denominator_inverses,
    unsigned int number_of_columns
) {
    unsigned int row_index = blockIdx.x * blockDim.x + threadIdx.x;

    for(unsigned int constraint_index = 0; constraint_index < number_of_columns - 2; constraint_index++) {
        numerators[row_index] = add(
            numerators[row_index],
            mul(
                sub(
                    add(
                        square(trace_evaluations[constraint_index][row_index]),
                        square(trace_evaluations[constraint_index + 1][row_index])
                    ),
                    trace_evaluations[constraint_index + 2][row_index]
                ),
                random_coeff_powers[number_of_columns - 3 - constraint_index]
            )
        );
    }

    qm31 constraint_quotient = mul(
        denominator_inverses[row_index],
        numerators[row_index]
    );

    quotients_0[row_index] = constraint_quotient.a.a;
    quotients_1[row_index] = constraint_quotient.a.b;
    quotients_2[row_index] = constraint_quotient.b.a;
    quotients_3[row_index] = constraint_quotient.b.b;
}

void evaluate_wide_fibonacci_constraint_quotients_on_domain(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 **trace_evaluations,
    qm31 *random_coeff_powers,
    m31 *denominator_inverses,
    unsigned int extended_domain_size,
    unsigned int number_of_columns
) {
    m31 **device_trace_evaluations = clone_to_device<m31*>(trace_evaluations, extended_domain_size);
    qm31 *numerators = (qm31*) cuda_alloc_zeroes_uint32_t(4 * extended_domain_size);

    unsigned int block_size = 1024;
    unsigned int number_of_blocks = (extended_domain_size + block_size - 1) / block_size;

    evaluate_wide_fibonacci_constraint_quotients_kernel<<<block_size, number_of_blocks>>>(
        quotients_0, quotients_1, quotients_2, quotients_3,
        device_trace_evaluations,
        numerators,
        random_coeff_powers,
        denominator_inverses,
        number_of_columns
    );
    cudaDeviceSynchronize();

    cuda_free_memory(device_trace_evaluations);
    cuda_free_memory(numerators);
}