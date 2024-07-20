#ifndef BIT_REVERSE_H
#define BIT_REVERSE_H

#include "fields.cuh"

extern "C"
void bit_reverse_base_field(uint32_t *, int, int);


extern "C"
void bit_reverse_secure_field(qm31 *, int, int);

#endif // BIT_REVERSE_H