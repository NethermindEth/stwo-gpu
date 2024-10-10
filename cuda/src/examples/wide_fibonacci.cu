#include "../../include/examples/wide_fibonacci.cuh"

__global__ void evaluate_wide_fibonacci_constraint_quotients_kernel(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 **trace_evaluations,
    qm31 *numerators,
    qm31 *random_coeff_powers,
    m31 *denominator_inverses,
    unsigned int extended_domain_size,
    unsigned int number_of_columns
) {
    // // Calculate f
    // let mut numerators = vec![SecureField::zero(); 1 << (ext_domain_log_size)];
    // let [mut accum] =
    //     evaluation_accumulator.columns([(ext_domain_log_size, self.n_constraints())]);

    // for i in 0..ext_domain.size() {
    //     // Step constraints.
    //     for j in 0..self.n_columns() - 2 {
    //         numerators[i] += accum.random_coeff_powers[self.n_columns() - 3 - j]
    //             * (trace_evals_ext_domain[0][j][i].square() + trace_evals_ext_domain[0][j + 1][i].square()
    //                 - trace_evals_ext_domain[0][j + 2][i]);
    //     }
    // }

    // // calculate t
    // for (i, (num, denom)) in numerators.iter().zip(denom_inverses.iter()).enumerate() {
    //     accum.accumulate(i, *num * *denom);
    // }

    for(unsigned int row_index = 0; row_index < extended_domain_size; row_index++) {
        for(unsigned int column_index = 0; column_index < number_of_columns - 2; column_index++) {
            numerators[row_index] = add(
                numerators[row_index],
                mul(
                    sub(
                        add(
                            square(trace_evaluations[column_index][row_index]),
                            square(trace_evaluations[column_index + 1][row_index])
                        ),
                        trace_evaluations[column_index + 2][row_index]
                    ),
                    random_coeff_powers[number_of_columns - 3 - column_index]
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

    // evaluate_wide_fibonacci_constraint_quotients_kernel<<<block_size, number_of_blocks>>>(
    evaluate_wide_fibonacci_constraint_quotients_kernel<<<1, 1>>>(
        quotients_0, quotients_1, quotients_2, quotients_3,
        device_trace_evaluations,
        numerators,
        random_coeff_powers,
        denominator_inverses,
        extended_domain_size,
        number_of_columns
    );
    cudaDeviceSynchronize();

    cuda_free_memory(device_trace_evaluations);
}