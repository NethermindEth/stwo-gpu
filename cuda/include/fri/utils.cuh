#ifndef FRI_UTILS_H
#define FRI_UTILS_H

#include "fields.cuh"

__device__ const qm31 getEvaluation(const uint32_t *const *eval_values, const uint32_t index);

#endif // FRI_UTILS_H