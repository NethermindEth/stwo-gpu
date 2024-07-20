#ifndef BIT_REVERSE_H
#define BIT_REVERSE_H

#include "utils.cuh"


typedef struct {
    uint32_t a;
    uint32_t b;
} cm31;

typedef struct {
    cm31 a;
    cm31 b;
} qm31;

extern "C"
void bit_reverse_base_field(uint32_t *, int, int);


extern "C"
void bit_reverse_secure_field(qm31 *, int, int);

#endif // BIT_REVERSE_H