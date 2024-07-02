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

////////////


extern "C"
__global__ void sum(uint32_t *from, uint32_t* temp, uint32_t *results, int size) {
    int idx = threadIdx.x;
    from = &from[blockIdx.x * blockDim.x];
    temp = &temp[blockIdx.x * blockDim.x];
    int data_size = size;
    size = min(size, 2048);
    
    temp[idx] = m31_sub(from[idx], from[idx + (data_size >> 1)]);
    size >>= 1;

    __syncthreads();

    while(size > 1) {
        if (idx < size) {
            temp[idx] = m31_add(temp[idx], temp[idx + (size >> 1)]);
        }
        
        __syncthreads();
    
        size >>= 1;
    }

    if(threadIdx.x == 0) {
        results[blockIdx.x] = temp[0];
    }
}

extern "C"
__global__ void pairwise_sum(uint32_t *from, uint32_t* temp, uint32_t *result, int size) {
    int idx = threadIdx.x;
    from = &from[blockIdx.x * blockDim.x];
    temp = &temp[blockIdx.x * blockDim.x];
    size = min(size, 2048);

    temp[idx] = m31_add(from[idx], from[idx + (size >> 1)]);
    size >>= 1;

    __syncthreads();

    while(size > 1) {
        if (idx < size) {
            temp[idx] = m31_add(temp[idx], temp[idx + (size >> 1)]);
        }
        
        __syncthreads();  // Can this be optimized if it's inside the if statement?
    
        size >>= 1;
    }

    if(blockIdx.x == 0 && threadIdx.x == 0) {
        result[0] = temp[0];
    }
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
