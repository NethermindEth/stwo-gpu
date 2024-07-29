#ifndef BLAKE2S_H
#define BLAKE2S_H

#include "fields.cuh"

extern "C"
void commit_on_layer(uint32_t size, uint32_t *column, char* result);

#endif // BLAKE2S_H