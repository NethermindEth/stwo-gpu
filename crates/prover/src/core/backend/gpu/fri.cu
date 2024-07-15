typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

typedef struct {
    uint32_t a;
    uint32_t b;
} cm31;

typedef struct {
    cm31 a;
    cm31 b;
} qm31;

const uint32_t P = 2147483647;
const cm31 R = {2, 1};

/*##### M31 ##### */

__device__ uint32_t m31_mul(uint32_t a, uint32_t b) {
    // TODO: use mul from m31.cu
    uint64_t v = ((uint64_t) a * (uint64_t) b);
    uint64_t w = v + (v >> 31);
    uint64_t u = v + (w >> 31);
    return u & P;}

__device__ uint32_t m31_add(uint32_t a, uint32_t b) {
    // TODO: use add from m31.cu
    return ((uint64_t) a + (uint64_t) b) % P;
}

__device__ uint32_t m31_sub(uint32_t a, uint32_t b) {
    // TODO: use sub from m31.cu
    return ((uint64_t) a + (uint64_t) (P - b)) % P;
}

__device__ uint32_t m31_neg(uint32_t a) {
    // TODO: use sub from m31.cu
    return P - a;
}


/*##### CM1 ##### */

__device__ cm31 cm31_mul(cm31 x, cm31 y) {
    return {m31_sub(m31_mul(x.a, y.a), m31_mul(x.b, y.b)), m31_add(m31_mul(x.a, y.b), m31_mul(x.b, y.a))};
}

__device__ cm31 cm31_add(cm31 x, cm31 y) {
    return {m31_add(x.a, y.a), m31_add(x.b, y.b)};
}

__device__ cm31 cm31_sub(cm31 x, cm31 y) {
    return {m31_sub(x.a, y.a), m31_sub(x.b, y.b)};
}

/*##### Q31 ##### */

__device__ qm31 qm31_mul(qm31 x, qm31 y) {
    return {
        cm31_add(
            cm31_mul(x.a, y.a),
            cm31_mul(R, cm31_mul(x.b, y.b))
        ),
        cm31_add(
            cm31_mul(x.a, y.b),
            cm31_mul(x.b, y.a)
        )
    };
}

__device__ qm31 qm31_add(qm31 x, qm31 y) {
    return {cm31_add(x.a, y.a), cm31_add(x.b, y.b)};
}

__device__ qm31 qm31_sub(qm31 x, qm31 y) {
    return {cm31_sub(x.a, y.a), cm31_sub(x.b, y.b)};
}

////////////

extern "C"
__device__ void sum_reduce(uint32_t *list, uint32_t* temp_list, uint32_t *results, const uint32_t list_size, uint32_t (*first_reduce_operation)(uint32_t a, uint32_t b)) {
    const uint32_t block_thread_index = threadIdx.x;
    const uint32_t first_thread_in_block_index = blockIdx.x * blockDim.x;
    const uint32_t grid_thread_index = first_thread_in_block_index + block_thread_index;
    const uint32_t half_list_size = list_size >> 1;

    if (grid_thread_index < half_list_size) {
        uint32_t *list_to_sum_in_block = &temp_list[first_thread_in_block_index];
        uint32_t &thread_result = list_to_sum_in_block[block_thread_index];

        thread_result = first_reduce_operation(
            list[grid_thread_index],
            list[grid_thread_index + half_list_size]);

        __syncthreads();

        uint32_t list_to_sum_in_block_half_size = min(half_list_size, blockDim.x) >> 1;
        while(block_thread_index < list_to_sum_in_block_half_size) {
            thread_result = m31_add(
                thread_result, list_to_sum_in_block[block_thread_index + list_to_sum_in_block_half_size]);

            __syncthreads();

            list_to_sum_in_block_half_size >>= 1;
        }

        const bool is_first_thread_in_block = block_thread_index == 0;
        if (is_first_thread_in_block) {
            results[blockIdx.x] = thread_result;
        }
    }
}

extern "C"
__global__ void sum(uint32_t *list, uint32_t* temp_list, uint32_t *results, const uint32_t list_size) {
    sum_reduce(list, temp_list, results, list_size, m31_sub);
}

extern "C"
__global__ void pairwise_sum(uint32_t *list, uint32_t* temp_list, uint32_t *results, const uint32_t list_size) {
    sum_reduce(list, temp_list, results, list_size, m31_add);
}

extern "C"
__global__ void compute_g_values(uint32_t *f_values, uint32_t *results, uint32_t lambda, int size) {
    // Computes one coordinate of the QM31 g_values for the decomposition f = g + lambda * v_n at the first step of FRI.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < size) {
        if(idx < (size >> 1)) {
            results[idx] = m31_sub(f_values[idx], lambda);
        }
        if(idx >= (size >> 1)) {
            results[idx] = m31_add(f_values[idx], lambda);
        }
    }
}

extern "C"
__global__ void fold_line(
    uint32_t *domain,
    uint32_t twiddle_offset,
    uint32_t n,
    uint32_t *eval_values_0,
    uint32_t *eval_values_1,
    uint32_t *eval_values_2,
    uint32_t *eval_values_3,
    qm31 alpha,
    uint32_t *folded_values_0,
    uint32_t *folded_values_1,
    uint32_t *folded_values_2,
    uint32_t *folded_values_3
) {
    if (blockIdx.x == 0) {
        // TODO: must support list with length bigger than 2^10
        uint32_t i = threadIdx.x;
        if (i < n / 2) {
            uint32_t index_left = 2*i;
            uint32_t index_right = index_left+1;
            
            qm31 f_x = {{eval_values_0[index_left],
                         eval_values_1[index_left]},
                        {eval_values_2[index_left],
                         eval_values_3[index_left]}};
            qm31 f_x_minus = {{eval_values_0[index_right],
                               eval_values_1[index_right]},
                              {eval_values_2[index_right],
                               eval_values_3[index_right]}};
            uint32_t x_inverse = domain[i + twiddle_offset];

            qm31 f_0 = qm31_add(f_x, f_x_minus);
            qm31 f_1_dot_x = qm31_sub(f_x, f_x_minus);
            qm31 f_1 = {
                m31_mul(f_1_dot_x.a.a, x_inverse),
                m31_mul(f_1_dot_x.a.b, x_inverse),
                m31_mul(f_1_dot_x.b.a, x_inverse),
                m31_mul(f_1_dot_x.b.b, x_inverse),
            };

            qm31 f_prime = qm31_add(f_0, qm31_mul(alpha, f_1));

            folded_values_0[i] = f_prime.a.a;
            folded_values_1[i] = f_prime.a.b;
            folded_values_2[i] = f_prime.b.a;
            folded_values_3[i] = f_prime.b.b;
        }
    }
}
