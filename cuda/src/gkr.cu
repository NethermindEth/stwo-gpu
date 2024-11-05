#include <cuda_runtime.h>
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

__device__ Fraction<qm31> add_fraction(Fraction<m31> lhs, Fraction<m31> rhs) {
    qm31 numerator = add(mul(lhs.numerator, rhs.denominator), mul(rhs.numerator, lhs.denominator)); 
    qm31 denominator = mul(lhs.denominator, rhs.denominator); 
    return Fraction<qm31> {numerator, denominator}; 
}

__device__ Fraction<qm31> add_fraction(Fraction<qm31> lhs, Fraction<qm31> rhs) {
    qm31 numerator = add(mul(lhs.numerator, rhs.denominator), mul(rhs.numerator, lhs.denominator)); 
    qm31 denominator = mul(lhs.denominator, rhs.denominator); 
    return Fraction<qm31>(numerator, denominator); 
}

__device__ Fraction<qm31> add_reciprocal(Reciprocal<qm31> lhs, Reciprocal<qm31> rhs) {
    // `1/a + 1/b = (a + b)/(a * b)`
    return Fraction<qm31>(add(lhs.x, rhs.x), mul(lhs.x, rhs.x));
}

__global__ void next_grand_product_layer_kernel(qm31 *layer, uint32_t layer_size, qm31 *next_layer, uint32_t next_layer_size) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < next_layer_size) {
        next_layer[tid] = mul(layer[tid * 2], layer[tid * 2 + 1]);
    }
}

void next_grand_product_layer(qm31 *layer, uint32_t layer_size, qm31 *next_layer, uint32_t next_layer_size) {
    const unsigned int BLOCK_SIZE = 1024;
    const unsigned int NUM_BLOCKS = (next_layer_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    next_grand_product_layer_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(layer, layer_size, next_layer, next_layer_size); 
    cudaDeviceSynchronize();
}

// optimize(daniel): use uint4 built in?
__global__ void eval_grand_product_sum_kernel( 
    qm31 *eq_evals, 
    qm31 *input_layer, 
    uint32_t n_terms,      
    qm31 *eval_at_0,
    qm31 *eval_at_2
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // todo: specify size of shared memory
    __shared__ qm31 shared_eval_0[1024];
    __shared__ qm31 shared_eval_2[1024];

    shared_eval_0[threadIdx.x] = {0, 0, 0, 0};
    shared_eval_2[threadIdx.x] = {0, 0, 0, 0};
    __syncthreads();

    if (tid < n_terms) {
        qm31 inp_at_r0i0 = input_layer[tid * 2]; 
        qm31 inp_at_r0i1 = input_layer[tid * 2 + 1];
        qm31 inp_at_r1i0 = input_layer[(n_terms + tid) * 2];
        qm31 inp_at_r1i1 = input_layer[(n_terms + tid) * 2 + 1];

        qm31 inp_at_r2i0 = sub(add(inp_at_r1i0, inp_at_r1i0), inp_at_r0i0); // inp_at_r2i0
        qm31 inp_at_r2i1 = sub(add(inp_at_r1i1, inp_at_r1i1), inp_at_r0i1); // inp_at_r2i1

        qm31 prod_at_r2i = mul(inp_at_r2i0, inp_at_r2i1); // prod_at_r2i
        qm31 prod_at_r0i = mul(inp_at_r0i0, inp_at_r0i1); // prod_at_r0i

        shared_eval_0[threadIdx.x] = mul(eq_evals[tid], prod_at_r0i); // eq_eval_at_0i * prod_at_r0i
        shared_eval_2[threadIdx.x] = mul(eq_evals[tid], prod_at_r2i); // eq_eval_at_0i * prod_at_r2i
    }
    __syncthreads();

    // Perform intra-block reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_eval_0[threadIdx.x] = add(shared_eval_0[threadIdx.x], shared_eval_0[threadIdx.x + s]);
            shared_eval_2[threadIdx.x] = add(shared_eval_2[threadIdx.x], shared_eval_2[threadIdx.x + s]);
        }
        __syncthreads();
    }

    // Set global memory for each block id, grab the reduced shared thread id w.r.t. each block
    if (threadIdx.x == 0) {
        eval_at_0[blockIdx.x] = shared_eval_0[threadIdx.x];
        eval_at_2[blockIdx.x] = shared_eval_2[threadIdx.x];
    }
}

void eval_grand_product_sum(
    qm31 *eq_evals, 
    qm31 *input_layer, 
    uint32_t n_terms,      
    qm31 *eval_at_0,
    qm31 *eval_at_2
) {
    const unsigned int BLOCK_SIZE = 1024;
    const unsigned int NUM_BLOCKS = (n_terms + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Arrays for intra-block reduction
    qm31 *eval_at_0_temp_h = (qm31 *)malloc(sizeof(qm31) * NUM_BLOCKS);
    qm31 *eval_at_2_temp_h = (qm31 *)malloc(sizeof(qm31) * NUM_BLOCKS);
    qm31 *eval_at_0_temp_d;
    qm31 *eval_at_2_temp_d;

    cudaMalloc((void **)&eval_at_0_temp_d, sizeof(qm31) * NUM_BLOCKS);
    cudaMalloc((void **)&eval_at_2_temp_d, sizeof(qm31) * NUM_BLOCKS);
    
    size_t shared_mem_size = 2 * BLOCK_SIZE * sizeof(qm31);
    eval_grand_product_sum_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(
        eq_evals,
        input_layer,
        n_terms,
        eval_at_0_temp_d,
        eval_at_2_temp_d
    );
    cudaDeviceSynchronize();

    // Post intra-block reduction
    // todo(daniel): move to kernel
    cudaMemcpy(eval_at_0_temp_h, eval_at_0_temp_d, sizeof(qm31) * NUM_BLOCKS, cudaMemcpyDeviceToHost);
    cudaMemcpy(eval_at_2_temp_h, eval_at_2_temp_d, sizeof(qm31) * NUM_BLOCKS, cudaMemcpyDeviceToHost);

    for (int i = 1; i < NUM_BLOCKS; ++i) {
        eval_at_0_temp_h[0] = add(eval_at_0_temp_h[0], eval_at_0_temp_h[i]);
        eval_at_2_temp_h[0] = add(eval_at_2_temp_h[0], eval_at_2_temp_h[i]);
    }

    cudaMemcpy(eval_at_0, eval_at_0_temp_h, sizeof(qm31), cudaMemcpyHostToDevice);
    cudaMemcpy(eval_at_2, eval_at_2_temp_h, sizeof(qm31), cudaMemcpyHostToDevice);

    free(eval_at_0_temp_h);
    free(eval_at_2_temp_h);
    cudaFree(eval_at_0_temp_d);
    cudaFree(eval_at_2_temp_d);
}

__global__ void next_logup_generic_layer_kernel(
    qm31 *numerators, 
    qm31 *denominators, 
    uint32_t size, 
    qm31 *next_numerators, 
    qm31 *next_denominators, 
    uint32_t next_size
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < next_size) {
        Fraction<qm31> a = Fraction<qm31>(numerators[tid * 2], denominators[tid * 2]);
        Fraction<qm31> b = Fraction<qm31>(numerators[tid * 2 + 1], denominators[tid * 2 + 1]);

        Fraction<qm31> res = add_fraction(a, b);
        next_numerators[tid] = res.numerator; 
        next_denominators[tid] = res.denominator; 
    }
}

void next_logup_generic_layer(
    qm31 *numerators, 
    qm31 *denominators, 
    uint32_t size, 
    qm31 *next_numerators, 
    qm31 *next_denominators, 
    uint32_t next_size
) {
    const unsigned int BLOCK_SIZE = 1024;
    const unsigned int NUM_BLOCKS = (next_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    next_logup_generic_layer_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(numerators, denominators, size, next_numerators, next_denominators, next_size); 
    cudaDeviceSynchronize();
}

__global__ void eval_logup_generic_sum_kernel(
    qm31 *eq_evals,
    qm31 *numerators,
    qm31 *denominators,
    uint32_t n_terms,
    qm31 lambda,
    qm31 *eval_at_0,
    qm31 *eval_at_2
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // todo: specify size of shared memory
    __shared__ qm31 shared_eval_0[512];
    __shared__ qm31 shared_eval_2[512];

    shared_eval_0[threadIdx.x] = {0, 0, 0, 0};
    shared_eval_2[threadIdx.x] = {0, 0, 0, 0};
    __syncthreads();

    if (tid < n_terms) {
        qm31 inp_numer_at_r0i0 = numerators[tid * 2];
        qm31 inp_denom_at_r0i0 = denominators[tid * 2];
        qm31 inp_numer_at_r0i1 = numerators[tid * 2 + 1];
        qm31 inp_denom_at_r0i1 = denominators[tid * 2 + 1];
        qm31 inp_numer_at_r1i0 = numerators[(n_terms + tid) * 2];
        qm31 inp_denom_at_r1i0 = denominators[(n_terms + tid) * 2];
        qm31 inp_numer_at_r1i1 = numerators[(n_terms + tid) * 2 + 1];
        qm31 inp_denom_at_r1i1 = denominators[(n_terms + tid) * 2 + 1];

        qm31 inp_numer_at_r2i0 = sub(add(inp_numer_at_r1i0, inp_numer_at_r1i0), inp_numer_at_r0i0);
        qm31 inp_denom_at_r2i0 = sub(add(inp_denom_at_r1i0, inp_denom_at_r1i0), inp_denom_at_r0i0);
        qm31 inp_numer_at_r2i1 = sub(add(inp_numer_at_r1i1, inp_numer_at_r1i1), inp_numer_at_r0i1);
        qm31 inp_denom_at_r2i1 = sub(add(inp_denom_at_r1i1, inp_denom_at_r1i1), inp_denom_at_r0i1);

        Fraction<qm31> fraction_eval_0 = add_fraction(Fraction<qm31>(inp_numer_at_r0i0, inp_denom_at_r0i0), Fraction<qm31>(inp_numer_at_r0i1, inp_denom_at_r0i1));
        Fraction<qm31> fraction_eval_2 = add_fraction(Fraction<qm31>(inp_numer_at_r2i0, inp_denom_at_r2i0), Fraction<qm31>(inp_numer_at_r2i1, inp_denom_at_r2i1));

        shared_eval_0[threadIdx.x] = mul(eq_evals[tid], add(fraction_eval_0.numerator, mul(lambda, fraction_eval_0.denominator))); 
        shared_eval_2[threadIdx.x] = mul(eq_evals[tid], add(fraction_eval_2.numerator, mul(lambda, fraction_eval_2.denominator))); 

        // shared_eval_0[threadIdx.x] = inp_numer_at_r0i1;
        // shared_eval_2[threadIdx.x] = inp_denom_at_r0i1;
    }
    __syncthreads();

    // Perform intra-block reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_eval_0[threadIdx.x] = add(shared_eval_0[threadIdx.x], shared_eval_0[threadIdx.x + s]);
            shared_eval_2[threadIdx.x] = add(shared_eval_2[threadIdx.x], shared_eval_2[threadIdx.x + s]);
        }
        __syncthreads();
    }

    // Set global memory for each block id, grab the reduced shared thread id w.r.t. each block
    if (threadIdx.x == 0) {
        eval_at_0[blockIdx.x] = shared_eval_0[threadIdx.x];
        eval_at_2[blockIdx.x] = shared_eval_2[threadIdx.x];
    }

}

void eval_logup_generic_sum(
    qm31 *eq_evals,
    qm31 *numerators,
    qm31 *denominators,
    uint32_t n_terms,
    qm31 lambda,
    qm31 *eval_at_0,
    qm31 *eval_at_2
) {
    const unsigned int BLOCK_SIZE = 512;
    const unsigned int NUM_BLOCKS = (n_terms + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Arrays for intra-block reduction
    qm31 *eval_at_0_temp_h = (qm31 *)malloc(sizeof(qm31) * NUM_BLOCKS);
    qm31 *eval_at_2_temp_h = (qm31 *)malloc(sizeof(qm31) * NUM_BLOCKS);
    qm31 *eval_at_0_temp_d;
    qm31 *eval_at_2_temp_d;

    cudaMalloc((void **)&eval_at_0_temp_d, sizeof(qm31) * NUM_BLOCKS);
    cudaMalloc((void **)&eval_at_2_temp_d, sizeof(qm31) * NUM_BLOCKS);

    eval_logup_generic_sum_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(
        eq_evals,
        numerators,
        denominators, 
        n_terms,
        lambda,
        eval_at_0_temp_d,
        eval_at_2_temp_d
    );
     //   printf("ASDA\n\n\n");

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("eval_logup_generic_sum_kernel launch error: %s\n", cudaGetErrorString(err));
    }

    // Synchronize to catch any runtime errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("eval_logup_generic_sum_kernel execution error: %s\n", cudaGetErrorString(err));
    }

    // Post intra-block reduction
    cudaMemcpy(eval_at_0_temp_h, eval_at_0_temp_d, sizeof(qm31) * NUM_BLOCKS, cudaMemcpyDeviceToHost);
    cudaMemcpy(eval_at_2_temp_h, eval_at_2_temp_d, sizeof(qm31) * NUM_BLOCKS, cudaMemcpyDeviceToHost);

    // for (int i = 1; i < NUM_BLOCKS; ++i) {
    //     eval_at_0_temp_h[0] = add(eval_at_0_temp_h[0], eval_at_0_temp_h[i]);
    //     eval_at_2_temp_h[0] = add(eval_at_2_temp_h[0], eval_at_2_temp_h[i]);
    // }

    // printf("num: %u, %u, %u, %u", eval_at_0_temp_h[0].a.a, eval_at_0_temp_h[0].a.b, eval_at_0_temp_h[0].b.a, eval_at_0_temp_h[0].b.b);
    // printf("den: %u, %u, %u, %u", eval_at_2_temp_h[0].a.a, eval_at_2_temp_h[0].a.b, eval_at_2_temp_h[0].b.a, eval_at_2_temp_h[0].b.b);
    //     printf("ASDA\n\n\n");

    
    cudaMemcpy(eval_at_0, eval_at_0_temp_h, sizeof(qm31), cudaMemcpyHostToDevice);
    cudaMemcpy(eval_at_2, eval_at_2_temp_h, sizeof(qm31), cudaMemcpyHostToDevice);

    free(eval_at_0_temp_h);
    free(eval_at_2_temp_h);
    cudaFree(eval_at_0_temp_d);
    cudaFree(eval_at_2_temp_d);
}

__global__ void eval_logup_multiplicities_sum_kernel(
    qm31 *eq_evals,
    m31 *numerators,
    qm31 *denominators,
    uint32_t n_terms,
    qm31 lambda,
    qm31 *eval_at_0,
    qm31 *eval_at_2
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // todo: specify size of shared memory
    __shared__ qm31 shared_eval_0[512];
    __shared__ qm31 shared_eval_2[512];

    shared_eval_0[threadIdx.x] = {0, 0, 0, 0};
    shared_eval_2[threadIdx.x] = {0, 0, 0, 0};
    __syncthreads();

    if (tid < n_terms) {
        m31 inp_numer_at_r0i0 = numerators[tid * 2];
        qm31 inp_denom_at_r0i0 = denominators[tid * 2];
        m31 inp_numer_at_r0i1 = numerators[tid * 2 + 1];
        qm31 inp_denom_at_r0i1 = denominators[tid * 2 + 1];
        m31 inp_numer_at_r1i0 = numerators[(n_terms + tid) * 2];
        qm31 inp_denom_at_r1i0 = denominators[(n_terms + tid) * 2];
        m31 inp_numer_at_r1i1 = numerators[(n_terms + tid) * 2 + 1];
        qm31 inp_denom_at_r1i1 = denominators[(n_terms + tid) * 2 + 1];

        m31 inp_numer_at_r2i0 = sub(add(inp_numer_at_r1i0, inp_numer_at_r1i0), inp_numer_at_r0i0);
        qm31 inp_denom_at_r2i0 = sub(add(inp_denom_at_r1i0, inp_denom_at_r1i0), inp_denom_at_r0i0);
        m31 inp_numer_at_r2i1 = sub(add(inp_numer_at_r1i1, inp_numer_at_r1i1), inp_numer_at_r0i1);
        qm31 inp_denom_at_r2i1 = sub(add(inp_denom_at_r1i1, inp_denom_at_r1i1), inp_denom_at_r0i1);

        Fraction<qm31> fraction_eval_0 = add_fraction(Fraction<m31>(inp_numer_at_r0i0, inp_denom_at_r0i0), Fraction<m31>(inp_numer_at_r0i1, inp_denom_at_r0i1));
        Fraction<qm31> fraction_eval_2 = add_fraction(Fraction<m31>(inp_numer_at_r2i0, inp_denom_at_r2i0), Fraction<m31>(inp_numer_at_r2i1, inp_denom_at_r2i1));

        shared_eval_0[threadIdx.x] = mul(eq_evals[tid], add(fraction_eval_0.numerator, mul(lambda, fraction_eval_0.denominator))); 
        shared_eval_2[threadIdx.x] = mul(eq_evals[tid], add(fraction_eval_2.numerator, mul(lambda, fraction_eval_2.denominator))); 

        // shared_eval_0[threadIdx.x] = inp_numer_at_r0i1;
        // shared_eval_2[threadIdx.x] = inp_denom_at_r0i1;
    }
    __syncthreads();

    // Perform intra-block reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_eval_0[threadIdx.x] = add(shared_eval_0[threadIdx.x], shared_eval_0[threadIdx.x + s]);
            shared_eval_2[threadIdx.x] = add(shared_eval_2[threadIdx.x], shared_eval_2[threadIdx.x + s]);
        }
        __syncthreads();
    }

    // Set global memory for each block id, grab the reduced shared thread id w.r.t. each block
    if (threadIdx.x == 0) {
        eval_at_0[blockIdx.x] = shared_eval_0[threadIdx.x];
        eval_at_2[blockIdx.x] = shared_eval_2[threadIdx.x];
    }

}

void eval_logup_multiplicities_sum(
    qm31 *eq_evals,
    m31 *numerators,
    qm31 *denominators,
    uint32_t n_terms,
    qm31 lambda,
    qm31 *eval_at_0,
    qm31 *eval_at_2
) {
    const unsigned int BLOCK_SIZE = 512;
    const unsigned int NUM_BLOCKS = (n_terms + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Arrays for intra-block reduction
    qm31 *eval_at_0_temp_h = (qm31 *)malloc(sizeof(qm31) * NUM_BLOCKS);
    qm31 *eval_at_2_temp_h = (qm31 *)malloc(sizeof(qm31) * NUM_BLOCKS);
    qm31 *eval_at_0_temp_d;
    qm31 *eval_at_2_temp_d;

    cudaMalloc((void **)&eval_at_0_temp_d, sizeof(qm31) * NUM_BLOCKS);
    cudaMalloc((void **)&eval_at_2_temp_d, sizeof(qm31) * NUM_BLOCKS);

    eval_logup_multiplicities_sum_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(
        eq_evals,
        numerators,
        denominators, 
        n_terms,
        lambda,
        eval_at_0_temp_d,
        eval_at_2_temp_d
    );
     //   printf("ASDA\n\n\n");

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("eval_logup_multiplicities_sum launch error: %s\n", cudaGetErrorString(err));
    }

    // Synchronize to catch any runtime errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("eval_logup_multiplicities_sum execution error: %s\n", cudaGetErrorString(err));
    }

    // Post intra-block reduction
    cudaMemcpy(eval_at_0_temp_h, eval_at_0_temp_d, sizeof(qm31) * NUM_BLOCKS, cudaMemcpyDeviceToHost);
    cudaMemcpy(eval_at_2_temp_h, eval_at_2_temp_d, sizeof(qm31) * NUM_BLOCKS, cudaMemcpyDeviceToHost);

    // for (int i = 1; i < NUM_BLOCKS; ++i) {
    //     eval_at_0_temp_h[0] = add(eval_at_0_temp_h[0], eval_at_0_temp_h[i]);
    //     eval_at_2_temp_h[0] = add(eval_at_2_temp_h[0], eval_at_2_temp_h[i]);
    // }

    // printf("num: %u, %u, %u, %u", eval_at_0_temp_h[0].a.a, eval_at_0_temp_h[0].a.b, eval_at_0_temp_h[0].b.a, eval_at_0_temp_h[0].b.b);
    // printf("den: %u, %u, %u, %u", eval_at_2_temp_h[0].a.a, eval_at_2_temp_h[0].a.b, eval_at_2_temp_h[0].b.a, eval_at_2_temp_h[0].b.b);
    //     printf("ASDA\n\n\n");

    
    cudaMemcpy(eval_at_0, eval_at_0_temp_h, sizeof(qm31), cudaMemcpyHostToDevice);
    cudaMemcpy(eval_at_2, eval_at_2_temp_h, sizeof(qm31), cudaMemcpyHostToDevice);

    free(eval_at_0_temp_h);
    free(eval_at_2_temp_h);
    cudaFree(eval_at_0_temp_d);
    cudaFree(eval_at_2_temp_d);
}

__global__ void eval_logup_singles_sum_kernel(
    qm31 *eq_evals,
    qm31 *denominators,
    uint32_t n_terms,
    qm31 lambda,
    qm31 *eval_at_0,
    qm31 *eval_at_2
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // todo: specify size of shared memory
    __shared__ qm31 shared_eval_0[512];
    __shared__ qm31 shared_eval_2[512];

    shared_eval_0[threadIdx.x] = {0, 0, 0, 0};
    shared_eval_2[threadIdx.x] = {0, 0, 0, 0};
    __syncthreads();

    if (tid < n_terms) {
        qm31 inp_denom_at_r0i0 = denominators[tid * 2];
        qm31 inp_denom_at_r0i1 = denominators[tid * 2 + 1];
        qm31 inp_denom_at_r1i0 = denominators[(n_terms + tid) * 2];
        qm31 inp_denom_at_r1i1 = denominators[(n_terms + tid) * 2 + 1];

        qm31 inp_denom_at_r2i0 = sub(add(inp_denom_at_r1i0, inp_denom_at_r1i0), inp_denom_at_r0i0);
        qm31 inp_denom_at_r2i1 = sub(add(inp_denom_at_r1i1, inp_denom_at_r1i1), inp_denom_at_r0i1);

        Fraction<qm31> fraction_eval_0 = add_reciprocal(Reciprocal<qm31>(inp_denom_at_r0i0), Reciprocal<qm31>(inp_denom_at_r0i1));
        Fraction<qm31> fraction_eval_2 = add_reciprocal(Reciprocal<qm31>(inp_denom_at_r2i0), Reciprocal<qm31>(inp_denom_at_r2i1));

        shared_eval_0[threadIdx.x] = mul(eq_evals[tid], add(fraction_eval_0.numerator, mul(lambda, fraction_eval_0.denominator))); 
        shared_eval_2[threadIdx.x] = mul(eq_evals[tid], add(fraction_eval_2.numerator, mul(lambda, fraction_eval_2.denominator))); 

        // shared_eval_0[threadIdx.x] = inp_numer_at_r0i1;
        // shared_eval_2[threadIdx.x] = inp_denom_at_r0i1;
    }
    __syncthreads();

    // Perform intra-block reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_eval_0[threadIdx.x] = add(shared_eval_0[threadIdx.x], shared_eval_0[threadIdx.x + s]);
            shared_eval_2[threadIdx.x] = add(shared_eval_2[threadIdx.x], shared_eval_2[threadIdx.x + s]);
        }
        __syncthreads();
    }

    // Set global memory for each block id, grab the reduced shared thread id w.r.t. each block
    if (threadIdx.x == 0) {
        eval_at_0[blockIdx.x] = shared_eval_0[threadIdx.x];
        eval_at_2[blockIdx.x] = shared_eval_2[threadIdx.x];
    }

}

void eval_logup_singles_sum(
    qm31 *eq_evals,
    qm31 *denominators,
    uint32_t n_terms,
    qm31 lambda,
    qm31 *eval_at_0,
    qm31 *eval_at_2
) {
    const unsigned int BLOCK_SIZE = 512;
    const unsigned int NUM_BLOCKS = (n_terms + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Arrays for intra-block reduction
    qm31 *eval_at_0_temp_h = (qm31 *)malloc(sizeof(qm31) * NUM_BLOCKS);
    qm31 *eval_at_2_temp_h = (qm31 *)malloc(sizeof(qm31) * NUM_BLOCKS);
    qm31 *eval_at_0_temp_d;
    qm31 *eval_at_2_temp_d;

    cudaMalloc((void **)&eval_at_0_temp_d, sizeof(qm31) * NUM_BLOCKS);
    cudaMalloc((void **)&eval_at_2_temp_d, sizeof(qm31) * NUM_BLOCKS);

    eval_logup_singles_sum_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(
        eq_evals,
        denominators, 
        n_terms,
        lambda,
        eval_at_0_temp_d,
        eval_at_2_temp_d
    );
     //   printf("ASDA\n\n\n");

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("eval_logup_singles_sum launch error: %s\n", cudaGetErrorString(err));
    }

    // Synchronize to catch any runtime errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("eval_logup_singles_sum execution error: %s\n", cudaGetErrorString(err));
    }

    // Post intra-block reduction
    cudaMemcpy(eval_at_0_temp_h, eval_at_0_temp_d, sizeof(qm31) * NUM_BLOCKS, cudaMemcpyDeviceToHost);
    cudaMemcpy(eval_at_2_temp_h, eval_at_2_temp_d, sizeof(qm31) * NUM_BLOCKS, cudaMemcpyDeviceToHost);

    // for (int i = 1; i < NUM_BLOCKS; ++i) {
    //     eval_at_0_temp_h[0] = add(eval_at_0_temp_h[0], eval_at_0_temp_h[i]);
    //     eval_at_2_temp_h[0] = add(eval_at_2_temp_h[0], eval_at_2_temp_h[i]);
    // }

    // printf("num: %u, %u, %u, %u", eval_at_0_temp_h[0].a.a, eval_at_0_temp_h[0].a.b, eval_at_0_temp_h[0].b.a, eval_at_0_temp_h[0].b.b);
    // printf("den: %u, %u, %u, %u", eval_at_2_temp_h[0].a.a, eval_at_2_temp_h[0].a.b, eval_at_2_temp_h[0].b.a, eval_at_2_temp_h[0].b.b);
    //     printf("ASDA\n\n\n");

    
    cudaMemcpy(eval_at_0, eval_at_0_temp_h, sizeof(qm31), cudaMemcpyHostToDevice);
    cudaMemcpy(eval_at_2, eval_at_2_temp_h, sizeof(qm31), cudaMemcpyHostToDevice);

    free(eval_at_0_temp_h);
    free(eval_at_2_temp_h);
    cudaFree(eval_at_0_temp_d);
    cudaFree(eval_at_2_temp_d);
}

__global__ void next_logup_multiplicities_layer_kernel(
    m31 *numerators, 
    qm31 *denominators, 
    uint32_t size, 
    qm31 *next_numerators, 
    qm31 *next_denominators, 
    uint32_t next_size
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < next_size) {
        Fraction<m31> a = Fraction<m31>(numerators[tid * 2], denominators[tid * 2]);
        Fraction<m31> b = Fraction<m31>(numerators[tid * 2 + 1], denominators[tid * 2 + 1]);

        Fraction<qm31> res = add_fraction(a, b);
        next_numerators[tid] = res.numerator; 
        next_denominators[tid] = res.denominator; 
    }
}

__global__ void next_logup_singles_layer_kernel(
    qm31 *denominators, 
    uint32_t size, 
    qm31 *next_numerators, 
    qm31 *next_denominators, 
    uint32_t next_size
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < next_size) {
        Reciprocal<qm31> even = Reciprocal<qm31>(denominators[tid * 2]);
        Reciprocal<qm31> odd = Reciprocal<qm31>(denominators[tid * 2 + 1]);
        Fraction<qm31> res = add_reciprocal(even, odd);

        next_numerators[tid] = res.numerator; 
        next_denominators[tid] = res.denominator; 
    }
}

void next_logup_multiplicities_layer(
    m31 *numerators, 
    qm31 *denominators, 
    uint32_t size, 
    qm31 *next_numerators, 
    qm31 *next_denominators, 
    uint32_t next_size
) {
    const unsigned int BLOCK_SIZE = 1024;
    const unsigned int NUM_BLOCKS = (next_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    next_logup_multiplicities_layer_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(numerators, denominators, size, next_numerators, next_denominators, next_size); 
    cudaDeviceSynchronize();
}

void next_logup_singles_layer(
    qm31 *denominators, 
    uint32_t size, 
    qm31 *next_numerators, 
    qm31 *next_denominators, 
    uint32_t next_size
) {
    const unsigned int BLOCK_SIZE = 1024;
    const unsigned int NUM_BLOCKS = (next_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    next_logup_singles_layer_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(denominators, size, next_numerators, next_denominators, next_size); 
    cudaDeviceSynchronize();
}

