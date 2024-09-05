#ifndef MLE_H
#define MLE_H

#include "fields.cuh"

extern "C"
void fix_first_variable(m31 *evals, int evals_size, qm31 assignment, qm31* output_evals);

#endif // MLE_H