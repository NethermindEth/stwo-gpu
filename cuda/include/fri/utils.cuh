#ifndef FRI_UTILS_H
#define FRI_UTILS_H

#include "fields.cuh"

__device__ const qm31 getEvaluation(m31 **eval_values, const uint32_t index);

#endif // FRI_UTILS_H