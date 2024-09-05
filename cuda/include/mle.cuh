#ifndef MLE_H
#define MLE_H

#include "fields.cuh"

extern "C"
void fix_first_variable_basefield(m31 *evals, int evals_size, qm31 assignment, qm31* output_evals);

extern "C"
void fix_first_variable_securefield(qm31 *evals, int evals_size, qm31 assignment, qm31* output_evals);

#endif // MLE_H