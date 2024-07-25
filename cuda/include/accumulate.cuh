#ifndef ACCUMULATE_H
#define ACCUMULATE_H

#include "fields.cuh"

extern "C"
void accumulate(uint32_t size, uint32_t *left_column_0, uint32_t *left_column_1, uint32_t *left_column_2,
                uint32_t *left_column_3, uint32_t *right_column_0, uint32_t *right_column_1, uint32_t *right_column_2,
                uint32_t *right_column_3);

#endif // ACCUMULATE_H