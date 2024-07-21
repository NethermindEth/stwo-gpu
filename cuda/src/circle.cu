#include "../include/circle.cuh"
#include "../include/batch_inverse.cuh"
#include "../include/bit_reverse.cuh"
#include "../include/fields.cuh"
#include "../include/point.cuh"
#include "../include/utils.cuh"


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
    m31* twiddles;
    cudaMalloc((void**)&twiddles, sizeof(m31) * size);
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

void interpolate(m31 *values, m31 *inverse_twiddles_tree, int values_size) {
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

void evaluate(m31 *values, m31 *inverse_twiddles_tree, int values_size) {
    int block_dim = 256;
    int num_blocks = ((values_size >> 1) + block_dim - 1) / block_dim;

    int log_values_size = log_2(values_size);
    int layer_domain_size = 1;
    int layer_domain_offset = (values_size >> 1) - 2;
    int i = log_values_size - 1;
    while (i > 0) {
        rfft_line_part<<<num_blocks, block_dim>>>(values, inverse_twiddles_tree, values_size, layer_domain_size, layer_domain_offset, i);
        if (i > 1) {
            layer_domain_size <<= 1;
            layer_domain_offset -= layer_domain_size;
        }
        i -= 1;
    }

    rfft_circle_part<<<num_blocks, block_dim>>>(values, inverse_twiddles_tree, values_size);
    cudaDeviceSynchronize();
}


