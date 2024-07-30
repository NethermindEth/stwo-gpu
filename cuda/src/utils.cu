#include "../include/utils.cuh"

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

uint32_t* cuda_alloc_zeroes_uint32_t(int size) {
    uint32_t* device_ptr = cuda_malloc_uint32_t(size);
    cudaMemset(device_ptr, 0x00, sizeof(uint32_t) * size);
    return device_ptr;
}

void free_uint32_t_vec(uint32_t *device_ptr) {
    cudaFree(device_ptr);
}
