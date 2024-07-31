#ifndef ACCUMULATE_H
#define ACCUMULATE_H

#include "fields.cuh"

extern "C"
void accumulate(uint32_t size,
                uint32_t *left_columns[],
                uint32_t *right_columns[]);

#endif // ACCUMULATE_H