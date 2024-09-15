#include "utils.cuh"

#include <cstdio>

__device__ uint32_t bit_reverse(uint32_t n, int bits) {
    unsigned int reversed_n = __brev(n);
    return reversed_n >> (32 - bits);
}

__host__ int log_2(int value) {
    return __builtin_ctz(value);
}

void copy_uint32_t_vec_from_device_to_host(uint32_t *device_ptr, uint32_t *host_ptr, int size) {
    cuda_mem_copy_device_to_host<uint32_t>(device_ptr, host_ptr, size);
}

uint32_t* copy_uint32_t_vec_from_host_to_device(uint32_t *host_ptr, int size) {
    uint32_t* device_ptr = cuda_malloc<uint32_t>(size);
    cuda_mem_copy_host_to_device(host_ptr, device_ptr, size);
    return device_ptr;
}

void copy_uint32_t_vec_from_device_to_device(uint32_t *from, uint32_t *dst, int size) {
    cuda_mem_copy_device_to_device<uint32_t>(from, dst, size);
}

uint32_t* cuda_malloc_uint32_t(int size) {
    uint32_t* device_ptr = cuda_malloc<uint32_t>(size);
    return device_ptr;
}

Blake2sHash* cuda_malloc_blake_2s_hash(int size) {
    Blake2sHash* device_ptr = cuda_malloc<Blake2sHash>(size);
    return device_ptr;
}

__global__ void print_array(uint32_t *array, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < size) {
        printf("%d, ", array[idx]);
    }
}

uint32_t* cuda_alloc_zeroes_uint32_t(int size) {
    uint32_t* device_ptr = cuda_malloc_uint32_t(size);
    cudaMemset(device_ptr, 0x00, sizeof(uint32_t) * size);
    return device_ptr;
}

Blake2sHash* cuda_alloc_zeroes_blake_2s_hash(int size) {
    Blake2sHash* device_ptr = cuda_malloc_blake_2s_hash(size);
    cudaMemset(device_ptr, 0x00, sizeof(uint32_t) * size);
    return device_ptr;
}

Blake2sHash* copy_blake_2s_hash_vec_from_host_to_device(Blake2sHash *host_ptr, uint32_t size) {
    Blake2sHash* device_ptr = clone_to_device<Blake2sHash>(host_ptr, size);
    return device_ptr;
}

void copy_blake_2s_hash_vec_from_device_to_host(Blake2sHash *device_ptr, Blake2sHash *host_ptr, uint32_t size) {
    cuda_mem_copy_device_to_host<Blake2sHash>(device_ptr, host_ptr, size);
}

void copy_blake_2s_hash_vec_from_device_to_device(Blake2sHash *from, Blake2sHash *dst, int size) {
    cuda_mem_copy_device_to_device<Blake2sHash>(from, dst, size);
}

uint32_t** copy_device_pointer_vec_from_host_to_device(uint32_t** host_ptr, uint32_t size) {
    uint32_t** device_ptr = clone_to_device<uint32_t*>(host_ptr, size);
    return device_ptr;
}

void cuda_free_memory(void *device_ptr) {
    cudaError_t err = cudaFree(device_ptr);
    if (err != cudaSuccess) {
        printf("Error freeing memory: %s\n", cudaGetErrorString(err));
    }
}
