#ifndef UTILS_H
#define UTILS_H

#include "fields.cuh"

struct Blake2sHash {
    unsigned int s[8];
};

__device__ __forceinline__ uint32_t bit_reverse(uint32_t n, int bits) {
    unsigned int reversed_n = __brev(n);
    return reversed_n >> (32 - bits);
}

__host__ __forceinline__ int log_2(int value) {
    return __builtin_ctz(value);
}

extern "C"
void copy_uint32_t_vec_from_device_to_host(uint32_t *, uint32_t*, int);

extern "C"
uint32_t* copy_uint32_t_vec_from_host_to_device(uint32_t*, int);

extern "C"
void copy_uint32_t_vec_from_device_to_device(uint32_t *, uint32_t*, int);

extern "C"
uint32_t* cuda_malloc_uint32_t(int);

extern "C"
Blake2sHash* cuda_malloc_blake_2s_hash(int);

extern "C"
uint32_t* cuda_alloc_zeroes_uint32_t(int);

extern "C"
Blake2sHash* cuda_alloc_zeroes_blake_2s_hash(int);

extern "C"
void free_uint32_t_vec(uint32_t*);

extern "C"
Blake2sHash* copy_blake_2s_hash_from_host_to_device(Blake2sHash *host_ptr);

extern "C"
void copy_blake_2s_hash_from_device_to_host(Blake2sHash *device_ptr, Blake2sHash *host_ptr);

extern "C"
void free_blake_2s_hash(Blake2sHash* device_ptr);

extern "C"
Blake2sHash* copy_blake_2s_hash_vec_from_host_to_device(Blake2sHash *host_ptr, uint32_t size);

extern "C"
void copy_blake_2s_hash_vec_from_device_to_host(Blake2sHash *device_ptr, Blake2sHash *host_ptr, uint32_t size);

extern "C"
void copy_blake_2s_hash_vec_from_device_to_device(Blake2sHash *from, Blake2sHash *dst, int size);

extern "C"
void free_blake_2s_hash_vec(Blake2sHash* device_ptr);

extern "C"
uint32_t** copy_device_pointer_vec_from_host_to_device(uint32_t** host_ptr, uint32_t size);

extern "C"
void free_device_pointer_vec(unsigned int **device_ptr);

#endif // UTILS_H
