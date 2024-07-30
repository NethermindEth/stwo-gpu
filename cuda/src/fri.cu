#include "../include/fri.cuh"
#include "../include/utils.cuh"
#include "../include/circle.cuh"

__device__ void sum_block_list(uint32_t *results, const uint32_t block_thread_index, const uint32_t half_list_size,
                               const uint32_t *list_to_sum_in_block, uint32_t &thread_result) {
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

__global__ void sum_reduce(uint32_t *list, uint32_t *temp_list, uint32_t *results, const uint32_t list_size) {
    const uint32_t block_thread_index = threadIdx.x;
    const uint32_t first_thread_in_block_index = blockIdx.x * blockDim.x;
    const uint32_t grid_thread_index = first_thread_in_block_index + block_thread_index;
    const uint32_t half_list_size = list_size >> 1;

    if (grid_thread_index < half_list_size) {
        uint32_t *list_to_sum_in_block = &temp_list[first_thread_in_block_index];
        uint32_t &thread_result = list_to_sum_in_block[block_thread_index];

        thread_result = sub(
                list[grid_thread_index],
                list[grid_thread_index + half_list_size]);

        __syncthreads();

        sum_block_list(results, block_thread_index, half_list_size, list_to_sum_in_block, thread_result);
    }
}

__global__ void sum_reduce2(uint32_t *list, uint32_t *temp_list, uint32_t *results, const uint32_t list_size) {
    const uint32_t block_thread_index = threadIdx.x;
    const uint32_t first_thread_in_block_index = blockIdx.x * blockDim.x;
    const uint32_t grid_thread_index = first_thread_in_block_index + block_thread_index;
    const uint32_t half_list_size = list_size >> 1;

    if (grid_thread_index < half_list_size) {
        uint32_t *list_to_sum_in_block = &temp_list[first_thread_in_block_index];
        uint32_t &thread_result = list_to_sum_in_block[block_thread_index];

        thread_result = add(
                list[grid_thread_index],
                list[grid_thread_index + half_list_size]);

        __syncthreads();

        sum_block_list(results, block_thread_index, half_list_size, list_to_sum_in_block, thread_result);
    }
}

extern "C"
void sum(uint32_t *list, const uint32_t list_size, uint32_t *result) {
    int block_dim = 1024;
    int num_blocks = (list_size / 2 + block_dim - 1) / block_dim;

    uint32_t *temp_list = cuda_malloc_uint32_t(list_size);
    uint32_t *results = cuda_alloc_zeroes_uint32_t(num_blocks);

    sum_reduce<<<num_blocks, min(list_size, block_dim)>>>(list, temp_list, results, list_size);
    cudaDeviceSynchronize();

    if (num_blocks == 1) {
        copy_uint32_t_vec_from_device_to_host(results, result, 1);
    } else {
        uint32_t *list_to_sum = cuda_malloc_uint32_t(num_blocks / 2 / block_dim);
        uint32_t *partial_results = results;
        uint32_t last_size = num_blocks;
        do {
            list_to_sum = partial_results;
            last_size = num_blocks;
            partial_results = list_to_sum;
            num_blocks = (num_blocks / 2 + block_dim - 1) / block_dim;

            sum_reduce2<<<num_blocks, block_dim>>>(
                    list_to_sum, temp_list, partial_results, last_size);
        } while (num_blocks > 1);
        copy_uint32_t_vec_from_device_to_host(partial_results, result, 1);
        free_uint32_t_vec(partial_results);
    }
    free_uint32_t_vec(temp_list);
    free_uint32_t_vec(results);
}

__global__ void compute_g_values_aux(uint32_t *f_values, uint32_t *results, int size, uint32_t lambda) {
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

void compute_g_values(uint32_t *f_values, uint32_t size, uint32_t lambda, uint32_t *g_value) {
    int block_dim = 1024;
    int num_blocks = (size + block_dim - 1) / block_dim;
    compute_g_values_aux<<<num_blocks, min(size, block_dim)>>>(f_values, g_value, size, lambda);
    cudaDeviceSynchronize();
}

qm31 sum_secure_field(uint32_t *column_0, uint32_t *column_1, uint32_t *column_2, uint32_t *column_3,
                      const uint32_t column_size) {
    m31 a, b, c, d;
    sum(column_0, column_size, &a);
    sum(column_1, column_size, &b);
    sum(column_2, column_size, &c);
    sum(column_3, column_size, &d);

    return {mul(a, inv(column_size)),
            mul(b, inv(column_size)),
            mul(c, inv(column_size)),
            mul(d, inv(column_size))};
}

void decompose(uint32_t *column_0, uint32_t *column_1, uint32_t *column_2, uint32_t *column_3, uint32_t size,
               qm31 *lambda, uint32_t *g_value_0, uint32_t *g_value_1, uint32_t *g_value_2,
               uint32_t *g_value_3) {
    *lambda = sum_secure_field(column_0, column_1, column_2, column_3, size);
    compute_g_values(column_0, size, lambda->a.a, g_value_0);
    compute_g_values(column_1, size, lambda->a.b, g_value_1);
    compute_g_values(column_2, size, lambda->b.a, g_value_2);
    compute_g_values(column_3, size, lambda->b.b, g_value_3);
}

__device__ uint32_t f(const uint32_t *domain,
                      const uint32_t twiddle_offset,
                      const uint32_t i) {
    return domain[i + twiddle_offset];
}

__device__ const qm31 getEvaluation(const uint32_t *const *eval_values, const uint32_t index) {
    return {{eval_values[0][index],
                    eval_values[1][index]},
            {eval_values[2][index],
                    eval_values[3][index]}};
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
               uint32_t *eval_values_1,
               uint32_t *eval_values_2,
               uint32_t *eval_values_3,
               uint32_t *eval_values_4,
               qm31 alpha,
               uint32_t *folded_values_1,
               uint32_t *folded_values_2,
               uint32_t *folded_values_3,
               uint32_t *folded_values_4) {
    int block_dim = 1024;
    int num_blocks = (n / 2 + block_dim - 1) / block_dim;
    fold_applying<<<num_blocks, block_dim>>>(
            gpu_domain,
            twiddle_offset,
            n,
            alpha,
            eval_values_1,
            eval_values_2,
            eval_values_3,
            eval_values_4,
            folded_values_1,
            folded_values_2,
            folded_values_3,
            folded_values_4);
    cudaDeviceSynchronize();
}

__device__ uint32_t g(uint32_t *domain,
                      uint32_t _twiddle_offset,
                      uint32_t i) {
    return get_twiddle(domain, i);
}

__global__ void fold_applying2(uint32_t *domain,
                               const uint32_t twiddle_offset,
                               const uint32_t n,
                               const qm31 alpha,
                               const qm31 alpha_sq,
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
    domain = &domain[twiddle_offset];

    if (i < n / 2) {
        const uint32_t x_inverse = g(domain, twiddle_offset, i);

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

void fold_circle_into_line(uint32_t *gpu_domain,
                           uint32_t twiddle_offset, uint32_t n,
                           uint32_t *eval_values_0,
                           uint32_t *eval_values_1,
                           uint32_t *eval_values_2,
                           uint32_t *eval_values_3,
                           qm31 alpha,
                           uint32_t *folded_values_0,
                           uint32_t *folded_values_1,
                           uint32_t *folded_values_2,
                           uint32_t *folded_values_3) {
    int block_dim = 1024;
    int num_blocks = (n / 2 + block_dim - 1) / block_dim;
    qm31 alpha_sq = mul(alpha, alpha);
    fold_applying2<<<num_blocks, block_dim>>>(
            gpu_domain,
            twiddle_offset,
            n,
            alpha,
            alpha_sq,
            eval_values_0,
            eval_values_1,
            eval_values_2,
            eval_values_3,
            folded_values_0,
            folded_values_1,
            folded_values_2,
            folded_values_3);
    cudaDeviceSynchronize();
}