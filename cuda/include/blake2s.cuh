#ifndef BLAKE2S_H
#define BLAKE2S_H

#include "fields.cuh"
#include "utils.cuh"

extern "C"
void commit_on_layer(uint32_t size, uint32_t *column, char* result);

extern "C"
void blake_2s_hash(uint32_t size, unsigned int *data, H *result);

#endif // BLAKE2S_H