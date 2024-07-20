#include "../include/utils.cuh"

__global__ void initialize_memory(int* device_array, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {
        device_array[tid] = tid;
    }
}

extern "C"
void copy_m31_vec_from_device_to_host(uint32_t *device_ptr, uint32_t *host_ptr, int size) {
    cudaMemcpy(host_ptr, device_ptr, sizeof(int) * size, cudaMemcpyDeviceToHost);
}

extern "C"
void free_uint32_t_vec(uint32_t *device_ptr) {
    cudaFree(device_ptr);
}

extern "C"
int* generate_array(int size) {
    int *device_array;
    cudaMalloc((void**)&device_array, size * sizeof(int));
    initialize_memory<<<256, 512>>>(device_array, size);
    cudaDeviceSynchronize();
    return device_array;
}

extern "C"
int sum(int *device_array, int size) {
    int* host_array = (int*)malloc(size * sizeof(int));
    cudaMemcpy(host_array, device_array, sizeof(int) * size, cudaMemcpyDeviceToHost);
    return host_array[size - 1];
}
