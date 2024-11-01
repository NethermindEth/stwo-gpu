#include "utils.cuh"
#include "gkr.cuh"

__global__ void gen_eq_evals_kernel(qm31 v, qm31 *factors, uint32_t y_size, qm31 *evals) {
    // Assumes `factors` holds 1 - y_i at position 2 * i and y_i at position 2 * i + 1
    // for all i = 0, .., y_size - 1.
    // TODO: See if shared memory speeds this up

    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    qm31 eq_eval = v;
    unsigned int shifted_thread_index = thread_index;
    for (int i = 2 * y_size - 2; i >= 0; i -= 2) {
        eq_eval = mul(eq_eval, factors[i + (shifted_thread_index & 1)]);
        shifted_thread_index >>= 1;
    }
    evals[thread_index] = eq_eval;
}

void gen_eq_evals(qm31 v, qm31 *y, uint32_t y_size, qm31 *evals, uint32_t evals_size) {
    const unsigned int BLOCK_SIZE = 1024;
    const unsigned int NUMBER_OF_BLOCKS = (evals_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int factors_byte_length = sizeof(qm31) * y_size * 2;
    qm31 *factors = (qm31*)malloc(factors_byte_length);
    for(int i = 0; i < y_size; i++) {
        factors[2 * i] = sub(m31{1}, y[i]);
        factors[2 * i + 1] = y[i];
    }

    qm31 *factors_device = clone_to_device<qm31>(factors, y_size * 2);
    free(factors);

    gen_eq_evals_kernel<<<NUMBER_OF_BLOCKS, min(evals_size, BLOCK_SIZE)>>>(v, factors_device, y_size, evals);
    cudaDeviceSynchronize();

    cudaFree(factors_device);
}

__host__ __device__ Fraction<qm31> add_fraction(Fraction<m31> lhs, Fraction<qm31> rhs) {
    qm31 numerator = add(mul(lhs.numerator, rhs.denominator), mul(rhs.numerator, lhs.denominator)); 
    qm31 denominator = mul(lhs.denominator, rhs.denominator); 
    return Fraction<qm31> {numerator, denominator}; 
}

__host__ __device__ Fraction<qm31> add_fraction(Fraction<qm31> lhs, Fraction<qm31> rhs) {
    qm31 numerator = add(mul(lhs.numerator, rhs.denominator), mul(rhs.numerator, lhs.denominator)); 
    qm31 denominator = mul(lhs.denominator, rhs.denominator); 
    return Fraction<qm31> {numerator, denominator}; 
}


__global__ void next_grand_product_layer_kernel(qm31 *layer, uint32_t layer_size, qm31 *next_layer, uint32_t next_layer_size) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < next_layer_size) {
        next_layer[tid] = mul(layer[tid * 2], layer[tid * 2 + 1]);
    }
}

void next_grand_product_layer(qm31 *layer, uint32_t layer_size, qm31 *next_layer, uint32_t next_layer_size) {
    const unsigned int BLOCK_SIZE = 1024;
    const unsigned int NUM_BLOCKS = (layer_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    uint32_t next_layer_size = layer_size / 2;
    qm31 *next_layer = (qm31 *)malloc(sizeof(qm31) * next_layer_size);
    next_grand_product_layer_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(layer, layer_size, next_layer, next_layer_size); 
    cudaDeviceSynchronize();
}