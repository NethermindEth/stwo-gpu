#ifndef UTILS_H
#define UTILS_H

#include "fields.cuh"

__device__ __forceinline__ unsigned int bit_reverse(unsigned int n, int bits) {
    unsigned int reversed_n = __brev(n);
    return reversed_n >> (32 - bits);
}

extern "C"
void copy_uint32_t_vec_from_device_to_host(uint32_t *, uint32_t*, int);

extern "C"
uint32_t* copy_uint32_t_vec_from_host_to_device(uint32_t*, int);

extern "C"
void free_uint32_t_vec(uint32_t*);

#endif // UTILS_H

