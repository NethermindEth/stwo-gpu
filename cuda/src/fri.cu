#include "../include/fri.cuh"
#include "../include/utils.cuh"
#include "../include/circle.cuh"

void
sum_reduce_with_first_reduce_operation(const m31 *list, const uint32_t list_size, const m31 *temp_list, m31 *results,
                                       m31 (*first_reduce_operation)(m31, m31));

uint32_t num_blocks_for(const uint32_t size) {
    uint32_t block_dim = max_block_dim;
    return (uint32_t) (size + block_dim - 1) / block_dim;
}

__device__ void sum_block_list(m31 *results,
                               const uint32_t block_thread_index,
                               const uint32_t half_list_size,
                               const m31 *list_to_sum_in_block,
                               m31 &thread_result) {
    uint32_t list_to_sum_in_block_half_size = min(half_list_size, blockDim.x) >> 1;
    while (block_thread_index < list_to_sum_in_block_half_size) {
        thread_result = add(
                thread_result, list_to_sum_in_block[block_thread_index + list_to_sum_in_block_half_size]);

        __syncthreads();

        list_to_sum_in_block_half_size >>= 1;
    }

    const bool is_first_thread_in_block = block_thread_index == 0;
    if (is_first_thread_in_block) {
        results[blockIdx.x] = thread_result;
    }
}

__device__ void
sum_reduce_with_first_reduce_operation(const m31 *list, const uint32_t list_size, m31 *temp_list, m31 *results,
                                       m31 (*first_reduce_operation)(m31, m31)) {
    const uint32_t block_thread_index = threadIdx.x;
    const uint32_t first_thread_in_block_index = blockIdx.x * blockDim.x;
    const uint32_t grid_thread_index = first_thread_in_block_index + block_thread_index;
    const uint32_t half_list_size = list_size >> 1;

    if (grid_thread_index < half_list_size) {
        m31 *list_to_sum_in_block = &temp_list[first_thread_in_block_index];
        m31 &thread_result = list_to_sum_in_block[block_thread_index];

        thread_result = first_reduce_operation(
                list[grid_thread_index],
                list[grid_thread_index + half_list_size]);

        __syncthreads();

        sum_block_list(results, block_thread_index, half_list_size, list_to_sum_in_block, thread_result);
    }
}

__global__ void sum_reduce(const m31 *list, m31 *temp_list, m31 *results, const uint32_t list_size) {
    sum_reduce_with_first_reduce_operation(list, list_size, temp_list, results, sub);
}

__global__ void sum_reduce2(const m31 *list, m31 *temp_list, m31 *results, const uint32_t list_size) {
    sum_reduce_with_first_reduce_operation(list, list_size, temp_list, results, add);
}

void get_vanishing_polynomial_coefficient(const m31 *list, const uint32_t list_size, m31 *result) {
    int num_blocks = num_blocks_for(list_size << 1);

    m31 *temp_list = cuda_malloc_uint32_t(list_size);
    m31 *results = cuda_alloc_zeroes_uint32_t(num_blocks);

    sum_reduce<<<num_blocks, min(list_size, max_block_dim)>>>(list, temp_list, results, list_size);
    cudaDeviceSynchronize();

    if (num_blocks == 1) {
        copy_uint32_t_vec_from_device_to_host(results, result, 1);
    } else {
        m31 *list_to_sum = cuda_malloc_uint32_t(num_blocks / 2 / max_block_dim);
        m31 *partial_results = results;
        uint32_t last_size = num_blocks;
        do {
            list_to_sum = partial_results;
            last_size = num_blocks;
            partial_results = list_to_sum;
            num_blocks = num_blocks_for(num_blocks / 2);

            sum_reduce2<<<num_blocks, max_block_dim>>>(
                    list_to_sum, temp_list, partial_results, last_size);
        } while (num_blocks > 1);
        copy_uint32_t_vec_from_device_to_host(partial_results, result, 1);
        free_uint32_t_vec(partial_results);
    }
    free_uint32_t_vec(temp_list);
    free_uint32_t_vec(results);
}

qm31 get_vanishing_polynomial_coefficients(const m31 *columns[4], const uint32_t column_size) {
    m31 a, b, c, d;
    get_vanishing_polynomial_coefficient(columns[0], column_size, &a);
    get_vanishing_polynomial_coefficient(columns[1], column_size, &b);
    get_vanishing_polynomial_coefficient(columns[2], column_size, &c);
    get_vanishing_polynomial_coefficient(columns[3], column_size, &d);

    return {mul(a, inv(column_size)),
            mul(b, inv(column_size)),
            mul(c, inv(column_size)),
            mul(d, inv(column_size))};
}

__global__ void compute_g_values(const m31 *f_values, m31 *results, int size, m31 lambda) {
    // Computes one coordinate of the QM31 g_values for the decomposition f = g + lambda * v_n at the first step of FRI.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (idx < (size >> 1)) {
            results[idx] = sub(f_values[idx], lambda);
        }
        if (idx >= (size >> 1)) {
            results[idx] = add(f_values[idx], lambda);
        }
    }
}


void decompose(const m31 *columns[4],
               uint32_t column_size,
               qm31 *lambda,
               uint32_t *g_values[4]) {
    *lambda = get_vanishing_polynomial_coefficients(columns, column_size);
    uint32_t num_blocks = num_blocks_for(column_size);
    uint32_t block_dim = min(column_size, max_block_dim);
    compute_g_values<<<num_blocks, block_dim>>>(
            columns[0], g_values[0], column_size, lambda->a.a);
    compute_g_values<<<num_blocks, block_dim>>>(
            columns[1], g_values[1], column_size, lambda->a.b);
    compute_g_values<<<num_blocks, block_dim>>>(
            columns[2], g_values[2], column_size, lambda->b.a);
    compute_g_values<<<num_blocks, block_dim>>>(
            columns[3], g_values[3], column_size, lambda->b.b);
    cudaDeviceSynchronize();
}

__device__ uint32_t f(const m31 *domain,
                      const uint32_t twiddle_offset,
                      const uint32_t i) {
    return domain[i + twiddle_offset];
}

__device__ const qm31 getEvaluation(const m31 *const *eval_values, const uint32_t index) {
    return {{eval_values[0][index],
                    eval_values[1][index]},
            {eval_values[2][index],
                    eval_values[3][index]}};
}

__global__ void fold_applying(const m31 *domain,
                              const uint32_t twiddle_offset,
                              const uint32_t n,
                              const qm31 alpha,
                              m31 *eval_values_0,
                              m31 *eval_values_1,
                              m31 *eval_values_2,
                              m31 *eval_values_3,
                              m31 *folded_values_0,
                              m31 *folded_values_1,
                              m31 *folded_values_2,
                              m31 *folded_values_3) {
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

void fold_line(m31 *gpu_domain,
               uint32_t twiddle_offset,
               uint32_t n,
               m31 *eval_values[4],
               qm31 alpha,
               m31 *folded_values[4]) {
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

__device__ uint32_t g(m31 *domain,
                      uint32_t _twiddle_offset,
                      uint32_t i) {
    return get_twiddle(domain, i);
}

__global__ void fold_applying2(m31 *domain,
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
                               m31 *folded_values_3) {
    const m31 *eval_values[4] = {eval_values_0,
                                 eval_values_1,
                                 eval_values_2,
                                 eval_values_3};

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    domain = &domain[twiddle_offset];

    if (i < n / 2) {
        const m31 x_inverse = g(domain, twiddle_offset, i);

        const uint32_t index_left = 2 * i;
        const uint32_t index_right = index_left + 1;

        const qm31 f_x = getEvaluation(eval_values, index_left);
        const qm31 f_x_minus = getEvaluation(eval_values, index_right);

        const qm31 f_0 = add(f_x, f_x_minus);
        const qm31 f_1 = mul_by_scalar(sub(f_x, f_x_minus), x_inverse);

        const qm31 f_prime = add(f_0, mul(alpha, f_1));

        qm31 previous_value = qm31{
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

void fold_circle_into_line(m31 *gpu_domain,
                           uint32_t twiddle_offset,
                           uint32_t n,
                           m31 *eval_values[4],
                           qm31 alpha,
                           m31 *folded_values[4]) {
    int block_dim = 1024;
    int num_blocks = (n / 2 + block_dim - 1) / block_dim;
    qm31 alpha_sq = mul(alpha, alpha);
    fold_applying2<<<num_blocks, block_dim>>>(
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