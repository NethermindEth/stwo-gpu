#ifndef FRI_H
#define FRI_H

#include "fields.cuh"

extern "C"
void fold_line(uint32_t *gpu_domain,
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

extern "C"
void fold_circle_into_line(uint32_t *gpu_domain,
                           uint32_t twiddle_offset, uint32_t n,
                           uint32_t *eval_values_0,
                           uint32_t *eval_values_1,
                           uint32_t *eval_values_2,
                           uint32_t *eval_values_3,
                           qm31 alpha,
                           uint32_t *folded_values_0,
                           uint32_t *folded_values_1,
                           uint32_t *folded_values_2,
                           uint32_t *folded_values_3);

extern "C"
void decompose(uint32_t *column_0, uint32_t *column_1, uint32_t *column_2, uint32_t *column_3, uint32_t size,
               qm31 *lambda, uint32_t *g_value_0, uint32_t *g_value_1, uint32_t *g_value_2,
               uint32_t *g_value_3);

#endif