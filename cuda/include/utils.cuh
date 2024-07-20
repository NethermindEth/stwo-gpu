#ifndef UTILS_H
#define UTILS_H

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

extern "C"
int* generate_array(int);

extern "C"
int last(int*, int);

extern "C"
void copy_uint32_t_vec_from_device_to_host(uint32_t *, uint32_t*, int);

extern "C"
uint32_t* copy_uint32_t_vec_from_host_to_device(uint32_t*, int);

extern "C"
void free_uint32_t_vec(uint32_t*);

#endif // UTILS_H

