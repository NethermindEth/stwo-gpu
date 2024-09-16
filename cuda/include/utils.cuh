#ifndef UTILS_H
#define UTILS_H

#include "fields.cuh"
#include <cstdio>

struct Blake2sHash {
    unsigned int s[8];
};

__device__ uint32_t bit_reverse(uint32_t n, int bits);

__host__ int log_2(int value);

extern "C"
void copy_uint32_t_vec_from_device_to_host(uint32_t *, uint32_t*, int);

extern "C"
void copy_uint32_t_vec_from_host_to_device(uint32_t *, uint32_t*, int);

extern "C"
void copy_uint32_t_vec_from_device_to_device(uint32_t *, uint32_t*, int);

extern "C"
uint32_t* clone_uint32_t_vec_from_host_to_device(uint32_t*, int);

extern "C"
uint32_t* cuda_malloc_uint32_t(int);

extern "C"
Blake2sHash* cuda_malloc_blake_2s_hash(int);

extern "C"
uint32_t* cuda_alloc_zeroes_uint32_t(int);

extern "C"
Blake2sHash* cuda_alloc_zeroes_blake_2s_hash(int);

extern "C"
void cuda_free_memory(void*);

template<typename T>
T* cuda_malloc(unsigned int size) {
    T *device_ptr;
    cudaError_t err = cudaMalloc((void**)&device_ptr, sizeof(T) * size);
    if (err != cudaSuccess) {
        printf("Error allocating memory: %s\n", cudaGetErrorString(err));
    } 
    return device_ptr;
}

template<typename T>
void cuda_mem_copy_host_to_device(T* host_data, T* device_data, unsigned int data_size) {
    cudaError_t err = cudaMemcpy(device_data, host_data, sizeof(T) * data_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Error copying memory: %s\n", cudaGetErrorString(err));
    } 
}

template<typename T>
void cuda_mem_copy_device_to_device(T* device_data_from, T* device_data_to, unsigned int data_size) {
    cudaError_t err = cudaMemcpy(device_data_to, device_data_from, sizeof(T) * data_size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        printf("Error copying memory: %s\n", cudaGetErrorString(err));
    } 
}

template<typename T>
void cuda_mem_copy_device_to_host(T* device_data, T* host_data, unsigned int data_size) {
    cudaError_t err = cudaMemcpy(host_data, device_data, sizeof(T) * data_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Error copying memory: %s\n", cudaGetErrorString(err));
    } 
}

template<typename T>
T* clone_to_device(T* host_data, unsigned int data_size) {
    T* device_data = cuda_malloc<T>(data_size);
    cuda_mem_copy_host_to_device(host_data, device_data, data_size);
    return device_data;
}


extern "C"
Blake2sHash* copy_blake_2s_hash_vec_from_host_to_device(Blake2sHash *host_ptr, uint32_t size);

extern "C"
void copy_blake_2s_hash_vec_from_device_to_host(Blake2sHash *device_ptr, Blake2sHash *host_ptr, uint32_t size);

extern "C"
void copy_blake_2s_hash_vec_from_device_to_device(Blake2sHash *from, Blake2sHash *dst, int size);

extern "C"
uint32_t** copy_device_pointer_vec_from_host_to_device(uint32_t** host_ptr, uint32_t size);

#endif // UTILS_H
