#ifndef CIRCLE_H
#define CIRCLE_H

#include "fields.cuh"
#include "point.cuh"

extern "C"
qm31 eval_at_point(m31 *coeffs, int coeffs_size, qm31 point_x, qm31 point_y);

#endif // CIRCLE_H