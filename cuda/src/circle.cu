#include "../include/circle.cuh"

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

m31* sort_values(m31 *from, int size) {
    int block_dim = 256;
    int num_blocks = (size + block_dim - 1) / block_dim;
    m31 *dst;
    cudaMalloc((void**)&dst, sizeof(m31) * size);
    sort_values_kernel<<<num_blocks, block_dim>>>(from, dst, size);
    cudaDeviceSynchronize();
    return dst;
}
