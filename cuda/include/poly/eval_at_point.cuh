#ifndef POLY_EVAL_AT_POINT_H
#define POLY_EVAL_AT_POINT_H

#include "../fields.cuh"

extern "C"
qm31 eval_at_point(m31 *coeffs, int coeffs_size, qm31 point_x, qm31 point_y);

#endif // POLY_EVAL_AT_POINT_H