#ifndef FRI_DECOMPOSE_H
#define FRI_DECOMPOSE_H

#include "fields.cuh"
#include "utils.cuh"

extern "C"
void decompose(uint32_t *columns[], uint32_t column_size, qm31 *lambda, uint32_t *g_values[]);

#endif // FRI_DECOMPOSE_H