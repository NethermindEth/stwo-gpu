#ifndef FRI_H
#define FRI_H

#include "fields.cuh"

extern "C"
void fold_line(m31 *gpu_domain,
               uint32_t twiddle_offset,
               uint32_t n,
               m31 *eval_values[4],
               qm31 alpha,
               m31 *folded_values[4]);

extern "C"
void fold_circle_into_line(m31 *gpu_domain,
                           uint32_t twiddle_offset,
                           uint32_t n,
                           m31 *eval_values[4],
                           qm31 alpha,
                           m31 *folded_values[4]);

extern "C"
void decompose(const m31 *columns[4],
               uint32_t column_size,
               qm31 *lambda,
               m31 *g_values[4]);

#endif