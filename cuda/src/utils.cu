#include "../include/utils.cuh"
#include <cstdio>

#include <cstdio>

void copy_uint32_t_vec_from_device_to_host(uint32_t *device_ptr, uint32_t *host_ptr, int size) {
    cudaMemcpy(host_ptr, device_ptr, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
}

uint32_t* copy_uint32_t_vec_from_host_to_device(uint32_t *host_ptr, int size) {
    uint32_t* device_ptr;
    cudaMalloc((void**)&device_ptr, sizeof(uint32_t) * size);
    cudaMemcpy(device_ptr, host_ptr, sizeof(uint32_t) * size, cudaMemcpyHostToDevice);
    return device_ptr;
}

void copy_uint32_t_vec_from_device_to_device(uint32_t *from, uint32_t *dst, int size) {
    cudaMemcpy(dst, from, sizeof(uint32_t) * size, cudaMemcpyDeviceToDevice);
}

uint32_t* cuda_malloc_uint32_t(int size) {
    uint32_t* device_ptr;
    cudaMalloc((void**)&device_ptr, sizeof(uint32_t) * size);
    return device_ptr;
}

Blake2sHash* cuda_malloc_blake_2s_hash(int size) {
    Blake2sHash* device_ptr;
    cudaMalloc((void**)&device_ptr, sizeof(Blake2sHash) * size);
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

void free_uint32_t_vec(uint32_t *device_ptr) {
    cudaError_t err = cudaFree(device_ptr);
    if (err != cudaSuccess) {
        printf("Error freeing memory: %s\n", cudaGetErrorString(err));
    }
}

Blake2sHash* copy_blake_2s_hash_from_host_to_device(Blake2sHash *host_ptr) {
    Blake2sHash* device_ptr;
    cudaMalloc((void**)&device_ptr, sizeof(Blake2sHash));
    cudaMemcpy(device_ptr, host_ptr, sizeof(Blake2sHash), cudaMemcpyHostToDevice);
    return device_ptr;
}

void copy_blake_2s_hash_from_device_to_host(Blake2sHash *device_ptr, Blake2sHash *host_ptr) {
    cudaMemcpy(host_ptr, device_ptr, sizeof(Blake2sHash), cudaMemcpyDeviceToHost);
}

void free_blake_2s_hash(Blake2sHash* device_ptr) {
    cudaFree(device_ptr);
}

Blake2sHash* copy_blake_2s_hash_vec_from_host_to_device(Blake2sHash *host_ptr, uint32_t size) {
    Blake2sHash* device_ptr;
    cudaMalloc((void**)&device_ptr, sizeof(Blake2sHash) * size);
    cudaMemcpy(device_ptr, host_ptr, sizeof(Blake2sHash) * size, cudaMemcpyHostToDevice);
    return device_ptr;
}

void copy_blake_2s_hash_vec_from_device_to_host(Blake2sHash *device_ptr, Blake2sHash *host_ptr, uint32_t size) {
    cudaMemcpy(host_ptr, device_ptr, sizeof(Blake2sHash) * size, cudaMemcpyDeviceToHost);
}

void copy_blake_2s_hash_vec_from_device_to_device(Blake2sHash *from, Blake2sHash *dst, int size) {
    cudaMemcpy(dst, from, sizeof(Blake2sHash) * size, cudaMemcpyDeviceToDevice);
}

void free_blake_2s_hash_vec(Blake2sHash* device_ptr) {
    cudaFree(device_ptr);
}

uint32_t** copy_device_pointer_vec_from_host_to_device(uint32_t** host_ptr, uint32_t size) {
    uint32_t** device_ptr;
    cudaError_t err = cudaMalloc((void**)&device_ptr, sizeof(uint32_t*) * size);
    if (err != cudaSuccess) {
        printf("Error allocating memory: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(device_ptr, host_ptr, sizeof(uint32_t*) * size, cudaMemcpyHostToDevice);
    return device_ptr;
}

void free_device_pointer_vec(unsigned int **device_ptr) {
    cudaError_t err = cudaFree(device_ptr);
    if (err != cudaSuccess) {
        printf("Error freeing memory: %s\n", cudaGetErrorString(err));
    }
}
