#include "poly/utils.cuh"

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

__global__ void rescale(m31 *values, int size, m31 factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        values[idx] = mul(values[idx], factor);
    }
}
