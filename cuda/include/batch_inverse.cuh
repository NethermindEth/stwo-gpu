#ifndef BATCH_INVERSE_H
#define BATCH_INVERSE_H

#include "fields.cuh"

extern "C"
void batch_inverse_base_field(uint32_t *from, uint32_t *dst, int size, int log_size);

#endif // BATCH_INVERSE_H