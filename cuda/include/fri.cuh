#ifndef FRI_H
#define FRI_H

#include "fields.cuh"

extern "C"
void fold_line(uint32_t *gpu_domain, uint32_t twiddle_offset, uint32_t n, uint32_t **eval_values, qm31 alpha,
               uint32_t **folded_values);

extern "C"
void fold_circle_into_line(uint32_t *gpu_domain,
                           uint32_t twiddle_offset,
                           uint32_t n,
                           uint32_t *eval_values[],
                           qm31 alpha,
                           uint32_t *folded_values[]);

extern "C"
void decompose(m31 *columns[],
               uint32_t column_size,
               qm31 *lambda,
               uint32_t *g_values[]);

#endif