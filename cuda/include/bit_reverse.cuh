#ifndef BIT_REVERSE_H
#define BIT_REVERSE_H

#include "fields.cuh"

extern "C"
void bit_reverse_base_field(m31*, int);


extern "C"
void bit_reverse_secure_field(qm31*, int);

template<typename T>
__global__ void bit_reverse_generic(T *array, int size, int bits);

#endif // BIT_REVERSE_H