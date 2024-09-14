#include "fri/utils.cuh"

__device__ const qm31 getEvaluation(m31 **eval_values, const uint32_t index) {
    return {{eval_values[0][index],
                    eval_values[1][index]},
            {eval_values[2][index],
                    eval_values[3][index]}};
}
