#ifndef ACCUMULATE_H
#define ACCUMULATE_H

#include "fields.cuh"

extern "C"
void accumulate(int size, m31 **left_columns, m31 **right_columns);

#endif // ACCUMULATE_H