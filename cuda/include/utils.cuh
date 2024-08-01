#ifndef UTILS_H
#define UTILS_H

#include "fields.cuh"
#include "quotients.cuh"

__device__ __forceinline__ uint32_t bit_reverse(uint32_t n, int bits) {
    unsigned int reversed_n = __brev(n);
    return reversed_n >> (32 - bits);
}

__host__ __forceinline__ int log_2(int value) {
    return __builtin_ctz(value);
}

extern "C"
void copy_uint32_t_vec_from_device_to_host(uint32_t *, uint32_t*, int);

extern "C"
uint32_t* copy_uint32_t_vec_from_host_to_device(uint32_t*, int);

extern "C"
void copy_uint32_t_vec_from_device_to_device(uint32_t *, uint32_t*, int);

extern "C"
uint32_t* cuda_malloc_uint32_t(int);

extern "C"
uint32_t* cuda_alloc_zeroes_uint32_t(int);

extern "C"
void free_uint32_t_vec(uint32_t*);

extern "C"
ColumnSampleBatch* copy_column_sample_batch_htd(ColumnSampleBatch *host_ptr, size_t size);

extern "C"
void free_column_sample_batch(ColumnSampleBatch* device_ptr, size_t size);

extern "C"
uint32_t** copy_column_circle_evaluation_htd(uint32_t *host_ptr, size_t column_size, size_t row_size);

extern "C"
void free_column_circle_evaluation(uint32_t** device_ptr, size_t size); 

extern "C" {
    uint32_t** unified_malloc_dbl_ptr_uint32_t(size_t size);
    void unified_set_dbl_ptr_uint32_t(uint32_t** h_out_ptr, uint32_t* d_in_ptr, size_t idx);
    QM31* copy_secure_field_vec_htd(QM31* host_ptr, int size);
    size_t* copy_size_t_vec_htd(size_t* host_ptr, int size);
    uint32_t** cuda_set_dbl_ptr_uint32_t(uint32_t** h_out_ptr, size_t size);
    void cuda_set_column_sample_batch(ColumnSampleBatch* device_ptr, CirclePoint<QM31> point, size_t* columns, QM31* values, size_t size, size_t idx);
    ColumnSampleBatch* cuda_malloc_column_sample_batch(size_t size);
    ColumnSampleBatch* copy_column_sample_batch_htd(ColumnSampleBatch *host_ptr, size_t size);
}

#endif // UTILS_H

