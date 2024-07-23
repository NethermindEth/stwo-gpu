#ifndef FRI_H
#define FRI_H

#include "fields.cuh"

extern "C"
uint32_t sum(uint32_t *list, const uint32_t list_size);

extern "C"
uint32_t *compute_g_values(uint32_t *f_values, uint32_t size, uint32_t lambda);

extern "C"
void fold_circle(uint32_t *gpu_domain,
                 uint32_t twiddle_offset, uint32_t n,
                 uint32_t *eval_values_1,
                 uint32_t *eval_values_2,
                 uint32_t *eval_values_3,
                 uint32_t *eval_values_4,
                 qm31 alpha,
                 uint32_t *folded_values_1,
                 uint32_t *folded_values_2,
                 uint32_t *folded_values_3,
                 uint32_t *folded_values_4);

#endif