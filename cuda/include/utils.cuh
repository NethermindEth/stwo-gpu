#ifndef EXAMPLE_H
#define EXAMPLE_H

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

extern "C"
int* generate_array(int);

extern "C"
int sum(int*, int);

extern "C"
void copy_m31_vec_from_device_to_host(uint32_t *, uint32_t*, int);

extern "C"
void free_uint32_t_vec(uint32_t*);

#endif // EXAMPLE_H

