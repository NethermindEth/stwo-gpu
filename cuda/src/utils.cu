#include "../include/utils.cuh"
#include "../include/quotients.cuh"

#include <cstdio>

void copy_uint32_t_vec_from_device_to_host(uint32_t *device_ptr, uint32_t *host_ptr, int size) {
    cudaMemcpy(host_ptr, device_ptr, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
}

uint32_t* copy_uint32_t_vec_from_host_to_device(uint32_t *host_ptr, int size) {
    uint32_t* device_ptr;
    cudaMalloc((void**)&device_ptr, sizeof(uint32_t) * size);
    cudaMemcpy(device_ptr, host_ptr, sizeof(uint32_t) * size, cudaMemcpyHostToDevice);
    return device_ptr;
}

void copy_uint32_t_vec_from_device_to_device(uint32_t *from, uint32_t *dst, int size) {
    cudaMemcpy(dst, from, sizeof(uint32_t) * size, cudaMemcpyDeviceToDevice);
}

uint32_t* cuda_malloc_uint32_t(int size) {
    uint32_t* device_ptr;
    cudaMalloc((void**)&device_ptr, sizeof(uint32_t) * size);
    return device_ptr;
}

__global__ void print_array(uint32_t *array, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < size) {
        printf("%d, ", array[idx]);
    }
}

uint32_t* cuda_alloc_zeroes_uint32_t(int size) {
    uint32_t* device_ptr = cuda_malloc_uint32_t(size);
    cudaMemset(device_ptr, 0x00, sizeof(uint32_t) * size);
    return device_ptr;
}

void free_uint32_t_vec(uint32_t *device_ptr) {
    cudaFree(device_ptr);
}

// Unified memory for device pointers
uint32_t** unified_malloc_dbl_ptr_uint32_t(size_t size) {
    uint32_t** unified_ptr;
    cudaMallocManaged(&unified_ptr, size * sizeof(uint32_t*));
    return unified_ptr; 
}

// Sets the index of the out pointer to the in pointer
void unified_set_dbl_ptr_uint32_t(uint32_t** h_out_ptr, uint32_t* d_in_ptr, size_t idx) {
   h_out_ptr[idx] = d_in_ptr;
}

ColumnSampleBatch* cuda_malloc_column_sample_batch(size_t size) {
    ColumnSampleBatch* device_ptr;
    cudaMalloc((void**)&device_ptr, size * sizeof(ColumnSampleBatch));
    return device_ptr; 
}

QM31* copy_secure_field_vec_htd(QM31* host_ptr, int size) {
    QM31* device_ptr;
    cudaMemcpy(host_ptr, device_ptr, size * sizeof(QM31), cudaMemcpyHostToDevice);
    return device_ptr; 
}

size_t* copy_size_t_vec_htd(size_t* host_ptr, int size) {
    size_t* device_ptr; 
    cudaMemcpy(host_ptr, device_ptr, size * sizeof(size_t), cudaMemcpyHostToDevice);
    return device_ptr; 
}

void cuda_set_column_sample_batch(ColumnSampleBatch* device_ptr, CirclePoint<QM31> point, size_t* columns, QM31* values, size_t size, size_t idx) {
    ColumnSampleBatch csb = ColumnSampleBatch{point, columns, values, size};
    device_ptr[idx] = csb; 
}

uint32_t** cuda_set_dbl_ptr_uint32_t(uint32_t** h_out_ptr, size_t size) {
    uint32_t** d_out_ptr; 
    cudaMalloc(&d_out_ptr, size * sizeof(uint32_t*));
    cudaMemcpy(d_out_ptr, h_out_ptr, size * sizeof(uint32_t*), cudaMemcpyHostToDevice);
    return d_out_ptr; 
}

ColumnSampleBatch* copy_column_sample_batch_htd(ColumnSampleBatch *host_ptr, size_t size) {
    ColumnSampleBatch* device_ptr;
    cudaMalloc((void**)&device_ptr, size * sizeof(ColumnSampleBatch));
    for(int i = 0; i < size; ++i) {
        ColumnSampleBatch* temp; 
        cudaMalloc((void**)&temp, sizeof(ColumnSampleBatch));

        // non-pointers
        temp->point = host_ptr[i].point;
        temp->size = host_ptr[i].size;

        // pointers
        cudaMalloc((void**)&temp->columns, host_ptr[i].size * sizeof(size_t));
        cudaMalloc((void**)&temp->values, host_ptr[i].size * sizeof(QM31));
        
        // copy 
        cudaMemcpy(temp->columns, host_ptr[i].columns, host_ptr[i].size * sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(temp->values, host_ptr[i].values, host_ptr[i].size * sizeof(QM31), cudaMemcpyHostToDevice);

        cudaMemcpy(&device_ptr[i], temp, sizeof(ColumnSampleBatch), cudaMemcpyHostToDevice);
    }
    return device_ptr;
}

void free_column_sample_batch(ColumnSampleBatch* device_ptr, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        ColumnSampleBatch* temp;
        cudaMemcpy(&temp, &device_ptr[i], sizeof(ColumnSampleBatch), cudaMemcpyDeviceToHost);
        cudaFree(temp->columns);
        cudaFree(temp->values);
    }   
    cudaFree(device_ptr);
}

uint32_t** copy_column_circle_evaluation_htd(uint32_t *host_ptr, size_t column_size, size_t row_size) {
    uint32_t** device_ptr; 
    cudaMalloc((void**)&device_ptr, column_size * sizeof(uint32_t*));
    for(int i = 0; i < column_size; ++i) {
        uint32_t* temp; 
        cudaMalloc((void**)&device_ptr, row_size * sizeof(uint32_t));
        cudaMemcpy(temp, &host_ptr[i], row_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    }
    return device_ptr; 
}


void free_column_circle_evaluation(uint32_t** device_ptr, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        uint32_t* temp;
        cudaMemcpy(temp, &device_ptr[i], sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaFree(temp);
    }   
    cudaFree(device_ptr);
}
