#include "../include/utils.cuh"

extern "C"
void copy_uint32_t_vec_from_device_to_host(uint32_t *device_ptr, uint32_t *host_ptr, int size) {
    cudaMemcpy(host_ptr, device_ptr, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
}

extern "C"
uint32_t* copy_uint32_t_vec_from_host_to_device(uint32_t *host_ptr, int size) {
    uint32_t* device_ptr;
    cudaMalloc((void**)&device_ptr, sizeof(uint32_t) * size);
    cudaMemcpy(device_ptr, host_ptr, sizeof(uint32_t) * size, cudaMemcpyHostToDevice);
    return device_ptr;
}

extern "C"
void free_uint32_t_vec(uint32_t *device_ptr) {
    cudaFree(device_ptr);
}
