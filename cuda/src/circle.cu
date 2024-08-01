#include "../include/circle.cuh"
#include "../include/batch_inverse.cuh"
#include "../include/bit_reverse.cuh"
#include "../include/fields.cuh"
#include "../include/point.cuh"
#include "../include/utils.cuh"
#include <assert.h>
#include <stdint.h>

__global__ void sort_values_kernel(m31 *from, m31 *dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if(idx < (size >> 1)) {
            dst[idx] = from[idx << 1];
        } else {
            int tmp = idx - (size >> 1);
            dst[idx] = from[size - (tmp << 1) - 1];
        }
    }
}

m31* sort_values_and_permute_with_bit_reverse_order(m31 *from, int size) {
    int block_dim = 256;
    int num_blocks = (size + block_dim - 1) / block_dim;
    m31 *dst;
    cudaMalloc((void**)&dst, sizeof(m31) * size);

    sort_values_kernel<<<num_blocks, block_dim>>>(from, dst, size);
    cudaDeviceSynchronize();

    bit_reverse_base_field(dst, size);
    return dst;
}

__global__ void precompute_twiddles_kernel(m31 *dst, point initial, point step, int offset, int size, int log_size) {
    // Computes one level of twiddles for a particular Coset.
    //      dst: twiddles array.
    //  initial: coset factor.
    //     step: generator of the group.
    //   offset: store values in dst[offset]
    //     size: coset size
    // log_size: log(size)

    // TODO: when size is larger than the max number of concurrent threads,
    //       consecutive numbers can be computed with a multiplication within the same thread,
    //       instead of using another pow.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    size = size >> 1;
    if (idx < size) {
        point pow = point_pow(step, bit_reverse(idx, log_size - 1)); // TODO: Be aware when log_size == 0 !
        dst[offset + idx] = point_mul(initial, pow).x;
    }
}

m31* precompute_twiddles(point initial, point step, int size) {
    m31* twiddles = cuda_malloc_uint32_t(size);
    m31 one = 1;
    cudaMemcpy(&twiddles[size - 1], &one, sizeof(m31), cudaMemcpyHostToDevice);
    int block_dim = 256;
    int num_blocks = (size + block_dim - 1) / block_dim;

    int log_size = log_2(size);

    // TODO: optimize
    int i = 0;
    int current_level_offset = 0;
    while (i < log_size) {
        precompute_twiddles_kernel<<<num_blocks, block_dim>>>(twiddles, initial, step, current_level_offset, size, log_2(size));
        initial = point_square(initial);
        step = point_square(step);
        size >>= 1;
        current_level_offset += size;
        i += 1;
    }
    cudaDeviceSynchronize();
    return twiddles;
}

__device__ int get_twiddle(m31 *twiddles, int index) {
    int k = index >> 2;
    if (index % 4 == 0) {
        return twiddles[2 * k + 1];
    } else if (index % 4 == 1) {
        return neg(twiddles[2 * k + 1]);
    } else if (index % 4 == 2) {
        return neg(twiddles[2 * k]);
    } else {
        return twiddles[2 * k];
    }
}

__global__ void ifft_circle_part(m31 *values, m31 *inverse_twiddles_tree, int values_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (values_size >> 1)) {
        m31 val0 = values[2 * idx];
        m31 val1 = values[2 * idx + 1];
        m31 twiddle = get_twiddle(inverse_twiddles_tree, idx);

        values[2 * idx] = add(val0, val1);
        values[2 * idx + 1] = mul(sub(val0, val1), twiddle);
    }
}


__global__ void ifft_line_part(m31 *values, m31 *inverse_twiddles_tree, int values_size, int inverse_twiddles_size, int layer_domain_offset, int layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (values_size >> 1)) {
        int number_polynomials = 1 << layer;
        int h = idx >> layer;
        int l = idx & (number_polynomials - 1);
        int idx0 = (h << (layer + 1)) + l;
        int idx1 = idx0 + number_polynomials;

        m31 val0 = values[idx0];
        m31 val1 = values[idx1];
        m31 twiddle = inverse_twiddles_tree[layer_domain_offset + h];
        
        values[idx0] = add(val0, val1);
        values[idx1] = mul(sub(val0, val1), twiddle);
    }
}

__global__ void rfft_circle_part(m31 *values, m31 *inverse_twiddles_tree, int values_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    
    if (idx < (values_size >> 1)) {
        m31 val0 = values[2 * idx];
        m31 val1 = values[2 * idx + 1];
        m31 twiddle = get_twiddle(inverse_twiddles_tree, idx);
        
        m31 temp = mul(val1, twiddle);
        
        values[2 * idx] = add(val0, temp);
        values[2 * idx + 1] = sub(val0, temp);
    }
}

__global__ void rfft_line_part(m31 *values, m31 *inverse_twiddles_tree, int values_size, int inverse_twiddles_size, int layer_domain_offset, int layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (values_size >> 1)) {
        int number_polynomials = 1 << layer;
        int h = idx / number_polynomials;
        int l = idx % number_polynomials;
        int idx0 = (h << (layer + 1)) + l;
        int idx1 = idx0 + number_polynomials;

        m31 val0 = values[idx0];
        m31 val1 = values[idx1];
        m31 twiddle = inverse_twiddles_tree[layer_domain_offset + h];
        
        m31 temp = mul(val1, twiddle);
        
        values[idx0] = add(val0, temp);
        values[idx1] = sub(val0, temp);
    }
}

__global__ void rescale(m31 *values, int size, m31 factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < size) {
        values[idx] = mul(values[idx], factor);
    }
}

void interpolate(int eval_domain_size, m31 *values, m31 *inverse_twiddles_tree, int inverse_twiddles_size, int values_size) {
    inverse_twiddles_tree = &inverse_twiddles_tree[inverse_twiddles_size - eval_domain_size];
    int block_dim = 256;
    int num_blocks = ((values_size >> 1) + block_dim - 1) / block_dim;
    ifft_circle_part<<<num_blocks, block_dim>>>(values, inverse_twiddles_tree, values_size);

    int log_values_size = log_2(values_size);
    int layer_domain_size = values_size >> 1;
    int layer_domain_offset = 0;
    int i = 1;
    while (i < log_values_size) {
        ifft_line_part<<<num_blocks, block_dim>>>(values, inverse_twiddles_tree, values_size, layer_domain_size, layer_domain_offset, i);

        layer_domain_size >>= 1;
        layer_domain_offset += layer_domain_size;
        i += 1;
    }
    cudaDeviceSynchronize();
        
    block_dim = 1024;
    num_blocks = (values_size + block_dim - 1) / block_dim;
    m31 factor = inv(pow(m31{ 2 }, log_values_size));
    rescale<<<num_blocks, block_dim>>>(values, values_size, factor);
    cudaDeviceSynchronize();
}

void evaluate(int eval_domain_size, m31 *values, m31 *twiddles_tree, int twiddles_size, int values_size) {
    twiddles_tree = &twiddles_tree[twiddles_size - eval_domain_size];
    int block_dim = 256;
    int num_blocks = ((values_size >> 1) + block_dim - 1) / block_dim;

    int log_values_size = log_2(values_size);
    int layer_domain_size = 1;
    int layer_domain_offset = (values_size >> 1) - 2;
    int i = log_values_size - 1;
    while (i > 0) {
        rfft_line_part<<<num_blocks, block_dim>>>(values, twiddles_tree, values_size, layer_domain_size, layer_domain_offset, i);
        layer_domain_size <<= 1;
        layer_domain_offset -= layer_domain_size;
        i -= 1;
    }

    rfft_circle_part<<<num_blocks, block_dim>>>(values, twiddles_tree, values_size);
    cudaDeviceSynchronize();
}

__global__ void eval_at_point_first_pass(m31* g_coeffs, qm31 *temp, qm31 *factors, int coeffs_size, int factors_size, int output_offset) {
    int idx = threadIdx.x;

    qm31 *output = &temp[output_offset];

    // Thread syncing happens within a block. 
    // Split the problem to feed them to multiple blocks.
    if(coeffs_size >= 512) {
        coeffs_size = 512;
    }
    
    extern __shared__ m31 s_coeffs[];
    extern __shared__ qm31 s_level[];

    s_coeffs[idx] = g_coeffs[2 * blockIdx.x * blockDim.x + idx];
    s_coeffs[idx + blockDim.x] = g_coeffs[2 * blockIdx.x * blockDim.x + idx + blockDim.x];
    __syncthreads();
    
    int level_size = coeffs_size >> 1;
    int factor_idx = factors_size - 1;

    if(idx < level_size) {
        m31 alpha = s_coeffs[2 * idx];
        m31 beta = s_coeffs[2 * idx + 1];
        qm31 factor = factors[factor_idx];

        qm31 result = { 
            {add(mul(beta, factor.a.a), alpha), mul(factor.a.b, beta)}, 
            {mul(beta,  factor.b.a), mul(beta, factor.b.b)} 
        };
        s_level[idx] = result;
    }
    factor_idx -= 1;
    level_size >>= 1;

    while(level_size > 0) {
        if(idx < level_size) {
            __syncthreads();
            qm31 a = s_level[2 * idx];
            qm31 b = s_level[2 * idx + 1];
            __syncthreads();
            s_level[idx] = add(a, mul(b, factors[factor_idx]));
        }
        factor_idx -= 1;
        level_size >>= 1;
        
    }

    if(idx == 0) {
        output[blockIdx.x] = s_level[0];
    }
}

__global__ void eval_at_point_second_pass(qm31* temp, qm31 *factors, int level_size, int factor_offset, int level_offset, int output_offset) {
    int idx = threadIdx.x;

    qm31 *level = &temp[level_offset];
    qm31 *output = &temp[output_offset];

    // Thread syncing happens within a block. 
    // Split the problem to feed them to multiple blocks.
    if(level_size >= 512) {
        level_size = 512;
    }
    
    extern __shared__ qm31 s_level[];

    s_level[idx] = level[2 * blockIdx.x * blockDim.x + idx];
    s_level[idx + blockDim.x] = level[2 * blockIdx.x * blockDim.x + idx + blockDim.x];

    level_size >>= 1;

    int factor_idx = factor_offset;

    while(level_size > 0) {
        if(idx < level_size) {
            __syncthreads();
            qm31 a = s_level[2 * idx];
            qm31 b = s_level[2 * idx + 1];
            __syncthreads();
            s_level[idx] = add(a, mul(b, factors[factor_idx]));
        }
        factor_idx -= 1;
        level_size >>= 1;
    }

    if(idx == 0) {
        output[blockIdx.x] = s_level[0];
    }
}

qm31 eval_at_point(m31 *coeffs, int coeffs_size, qm31 point_x, qm31 point_y) {
    int log_coeffs_size = log_2(coeffs_size);

    qm31 *host_mappings = (qm31*)malloc(sizeof(qm31) * log_coeffs_size);
    host_mappings[log_coeffs_size - 1] = point_y;
    host_mappings[log_coeffs_size - 2] = point_x;
    qm31 x = point_x;
    for(int i = 2; i < log_coeffs_size; i+=1) {
        x = sub(mul(qm31{cm31{2, 0}, cm31{0, 0}}, mul(x, x)), qm31{cm31{1, 0}, cm31{0, 0}});
        host_mappings[log_coeffs_size - 1 - i] = x;
    }

    int temp_memory_size = 0;
    int size = coeffs_size;
    while(size > 1) {
        size = (size + 511) / 512;
        temp_memory_size += size;
    }

    qm31* temp;
    cudaMalloc((void**)&temp, sizeof(qm31) * temp_memory_size);

    qm31* device_mappings;
    cudaMalloc((void**)&device_mappings, sizeof(qm31) * log_coeffs_size);
    cudaMemcpy(device_mappings, host_mappings, sizeof(qm31) * log_coeffs_size, cudaMemcpyHostToDevice);
    free(host_mappings);

    // First pass
    int block_dim = 256;
    int num_blocks = ((coeffs_size >> 1) + block_dim - 1) / block_dim;
    int shared_memory_bytes = 512 * 4 + 512 * 8;
    int output_offset = temp_memory_size - num_blocks;

    eval_at_point_first_pass<<<num_blocks, block_dim, shared_memory_bytes>>>(coeffs, temp, device_mappings, coeffs_size, log_coeffs_size, output_offset);

    // Second pass
    int mappings_offset = log_coeffs_size - 1;
    int level_offset = output_offset;
    while (num_blocks > 1) {
        mappings_offset -= 9;
        int new_num_blocks = ((num_blocks >> 1) + block_dim - 1) / block_dim;
        shared_memory_bytes = 512 * 4 * 4;
        output_offset = level_offset - new_num_blocks;
        eval_at_point_second_pass<<<new_num_blocks, block_dim, shared_memory_bytes>>>(temp, device_mappings, num_blocks, mappings_offset, level_offset, output_offset);
        num_blocks = new_num_blocks;
        level_offset = output_offset;
    }

    qm31 result = qm31{cm31{0,0}, cm31{0,1}};
    cudaDeviceSynchronize();
    cudaMemcpy(&result, temp, sizeof(qm31), cudaMemcpyDeviceToHost);
    cudaFree(temp);
    cudaFree(device_mappings);
    return result;
}


// CirclePoint Implementation
template<typename F>
__host__ __device__ 
CirclePoint<F>::CirclePoint() : x(F()), y(F()) {}

template<typename F>
__host__ __device__ 
CirclePoint<F>::CirclePoint(F x, F y) : x(x), y(y) {}

template<typename F>
__host__ __device__ constexpr 
CirclePoint<F> CirclePoint<F>::from_f(F x, F y) {
    return CirclePoint<F>(x, y);
}

template<typename F>
__host__ __device__ 
CirclePoint<F> CirclePoint<F>::conjugate() const {
    return CirclePoint(x, -y); 
}

template<typename F>
__host__ __device__ 
CirclePoint<F> CirclePoint<F>::antipode() const {
    return CirclePoint(-x, -y); 
}

template<typename F>
__host__ __device__ 
CirclePoint<F> CirclePoint<F>::zero() const {
    return CirclePoint(F::one(), F::zero());
}

template<typename F>
__host__ __device__ 
CirclePoint<F> CirclePoint<F>::double_val() const {
    return *this + *this;
}

template<typename F>
__host__ __device__ inline 
CirclePoint<F> CirclePoint<F>::operator+(const CirclePoint<F>& rhs) const {
    F x = x * rhs.x - y * rhs.y;
    F y = x * rhs.y + y * rhs.x;
    return CirclePoint(x, y);
}

template<typename F>
__host__ __device__ inline 
CirclePoint<F> CirclePoint<F>::operator-() const {
    return this->conjugate();
}

template<typename F>
__device__ CirclePoint<F> CirclePoint<F>::mul(unsigned long long scalar) const {
    CirclePoint res = CirclePoint::zero();
    CirclePoint cur = *this;  
    while (scalar > 0) {
        if (scalar & 1) {
            res = res + cur;
        }
        cur = cur.double_val();
        scalar >>= 1;
    }
    return res;
}

// CirclePointIndex
#define M31_CIRCLE_LOG_ORDER 31u

__host__ __device__ 
CirclePointIndex::CirclePointIndex(size_t idx) : idx(idx) {}

__host__ __device__ 
CirclePointIndex CirclePointIndex::zero() {
    return CirclePointIndex(0);
}

__host__ __device__ 
CirclePointIndex CirclePointIndex::generator() {
    return CirclePointIndex(1);
}

__device__ 
CirclePointIndex CirclePointIndex::reduce() const {
    return CirclePointIndex(idx & ((1 << M31_CIRCLE_LOG_ORDER) - 1));
}

__host__ __device__ 
CirclePointIndex CirclePointIndex::subgroup_gen(uint32_t log_size) {
    assert(log_size <= M31_CIRCLE_LOG_ORDER);
    return CirclePointIndex(1 << (M31_CIRCLE_LOG_ORDER - log_size));
}

// Move const away
__host__ __device__ 
CirclePoint<M31> CirclePointIndex::to_point() const {
    const CirclePoint<M31> M31_CIRCLE_GEN = CirclePoint<M31>(M31(2), M31(1268011823));
    return M31_CIRCLE_GEN.mul(static_cast<unsigned long long>(idx));
}

__host__ __device__ 
CirclePointIndex CirclePointIndex::half() const {
    assert((idx & 1) == 0);
    return CirclePointIndex(idx >> 1);
}

__host__ __device__
CirclePointIndex CirclePointIndex::operator+(const CirclePointIndex& rhs) const {
    return CirclePointIndex(idx + rhs.idx).reduce();
}

__host__ __device__ 
CirclePointIndex CirclePointIndex::operator-(const CirclePointIndex& rhs) const {
    return CirclePointIndex(idx + (1 << M31_CIRCLE_LOG_ORDER) - rhs.idx).reduce();
}

__host__ __device__ 
CirclePointIndex CirclePointIndex::operator*(size_t rhs) const {
    return CirclePointIndex(idx * rhs).reduce();
}

__host__ __device__ 
CirclePointIndex CirclePointIndex::operator-() const {
    return CirclePointIndex((1 << M31_CIRCLE_LOG_ORDER) - idx).reduce();
}

__host__ __device__ 
CirclePointIndex CirclePointIndex::mul(size_t rhs) const {
    return CirclePointIndex(idx * rhs).reduce(); 
}

// Coset
__host__ __device__ 
Coset::Coset(const CirclePointIndex& initial, const CirclePointIndex& step, uint32_t size) : initial_index(initial), step_size(step), log_size(size) {}

__host__ __device__ 
size_t Coset::size() const {
    return 1u << log_size;
}

__host__ __device__ 
CirclePointIndex Coset::index_at(size_t i) const {
    return initial_index + step_size.mul(i);
}

// Circle Domain
__host__ __device__ CircleDomain::CircleDomain(const Coset& half_coset) : half_coset(half_coset) {}

__host__ __device__ 
uint32_t CircleDomain::log_size() {
    return half_coset.log_size + 1; 
}
 
__host__ __device__ 
CirclePoint<M31> CircleDomain::at(size_t i) {
    return half_coset.index_at(i).to_point(); 
}

__host__ __device__ 
CirclePointIndex CircleDomain::index_at(size_t i) {
    if (i < half_coset.size()) {
        return half_coset.index_at(i);
    }
    else {
        return -half_coset.index_at(i - half_coset.size());
    }
}
