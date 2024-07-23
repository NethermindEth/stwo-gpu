#ifndef FRI_H
#define FRI_H

#include "fields.cuh"

extern "C"
uint32_t sum(uint32_t *list, const uint32_t list_size);

extern "C"
uint32_t* compute_g_values(uint32_t *f_values, uint32_t size, uint32_t lambda);

#endif