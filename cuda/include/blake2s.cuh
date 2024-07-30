#ifndef BLAKE2S_H
#define BLAKE2S_H

#include "fields.cuh"
#include "utils.cuh"

extern "C"
void commit_on_first_layer(uint32_t size, uint32_t *column, H* result);

#endif // BLAKE2S_H