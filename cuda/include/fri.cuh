#ifndef FRI_H
#define FRI_H

#include "fields.cuh"

extern "C"
void sum(uint32_t *list, uint32_t *temp_list, uint32_t *results, const uint32_t list_size);

#endif