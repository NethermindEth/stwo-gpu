#include "../include/circle.cuh"
#include "../include/batch_inverse.cuh"
#include "../include/bit_reverse.cuh"
#include "../include/fields.cuh"
#include "../include/point.cuh"
#include "../include/utils.cuh"

#include <assert.h>
#include <stdint.h>
#include <cstdio>

__device__ 
QM31 accumulate_row_quotients(
    uint32_t* values, 
    ColumnSampleBatch* sample_batches,
    size_t sample_batches_size, 
    uint32_t** columns, 
    size_t columns_size,
    size_t columns_row_size,
    QM31** d_denominator_inverses,
    QM31* random_coeffs,
    size_t row
);

__global__
void accumulate_quotients_helper (
    uint32_t* values, 
    ColumnSampleBatch* sample_batches,
    size_t sample_batches_size, 
    CircleDomain domain,
    size_t domain_size,
    uint32_t** columns, 
    size_t columns_size,
    size_t columns_row_size,
    QM31** d_denominator_inverses,
    QM31* random_coeffs
);

__global__ void 
batch_random_coeffs(    
    QM31* res, 
    ColumnSampleBatch* sample_batches,
    size_t sample_batches_size, 
    QM31 random_coeff
);
__device__ void 
point_vanishing_fraction(
    CirclePoint<QM31> vanish_point, 
    CirclePoint<M31> p, 
    QM31* num, 
    QM31* denom
);
QM31** denominator_inverses(
    ColumnSampleBatch* sample_batches,
    size_t sample_batches_size, 
    CircleDomain domain,
    size_t domain_size
);
__global__ void 
enumerate_fractions(
    QM31* flat_denominators,
    QM31* numerator_terms,
    ColumnSampleBatch* sample_batches,
    size_t sample_batches_size, 
    CircleDomain domain,
    size_t domain_size, 
    size_t n_fractions
);
__global__ void denominator_inverses_mul_numerator(QM31* flat_denominator_inverses, QM31* numerator_terms, size_t n_fractions);
__global__ void denominator_inverses_bit_reverse(QM31** denominator_inverses, QM31* flat_denominator_inverses, size_t domain_size);
__global__ void printColumnSampleBatchKernel(const ColumnSampleBatch* batch, CircleDomain domain);
__global__ void printqm31(const QM31* batch, size_t size);
__global__ void print_values(uint32_t** values, size_t domain_size);
__global__ void print_values_test(uint32_t* values, size_t domain_size);
// ---------------------------------------------------------------------------------------------------------
const uint32_t EXTENSION_DEGREE = 4; 

__global__
void initialize_device_ptr_return_value(uint32_t** out_arr, uint32_t* in_arr1, uint32_t* in_arr2, uint32_t* in_arr3, uint32_t* in_arr4) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid == 0) {
        out_arr[0] = in_arr1;
        out_arr[1] = in_arr2;
        out_arr[2] = in_arr3;
        out_arr[3] = in_arr4;
    }
}

uint32_t** initialize_values(size_t domain_size) {
    uint32_t** values; 
    uint32_t* in_arr1, * in_arr2, * in_arr3, * in_arr4; 
    cudaMalloc((void**)&values, 4 * sizeof(uint32_t*));
    cudaMalloc((void**)&in_arr1, domain_size * sizeof(uint32_t)); 
    cudaMalloc((void**)&in_arr2, domain_size * sizeof(uint32_t)); 
    cudaMalloc((void**)&in_arr3, domain_size * sizeof(uint32_t)); 
    cudaMalloc((void**)&in_arr4, domain_size * sizeof(uint32_t)); 
    cudaMemset(in_arr1, 0, domain_size); 
    cudaMemset(in_arr1, 1, domain_size); 
    cudaMemset(in_arr1, 2, domain_size); 
    cudaMemset(in_arr1, 3, domain_size); 
    initialize_device_ptr_return_value<<<1, 1>>>(values, in_arr1, in_arr2, in_arr3, in_arr4);
    return values; 
}


uint32_t* accumulate_quotients(
    uint32_t* value_columns1,
    uint32_t* value_columns2,
    uint32_t* value_columns3,
    uint32_t* value_columns4,
    size_t domain_initial_index,
    size_t domain_step_size,
    uint32_t domain_log_size, 
    size_t domain_size, 
    uint32_t** columns, 
    size_t columns_size,
    size_t columns_row_size,
    QM31 random_coeff,
    ColumnSampleBatch* sample_batches,
    size_t sample_batches_size 
) {
    // Return Value
    //uint32_t** values = initialize_values(domain_size); 
    uint32_t* values;
    cudaMalloc((void**)&values, EXTENSION_DEGREE * domain_size * sizeof(uint32_t));    // flatten 2d array

    int threads_per_block = 1024; 
    int blocks_per_grid = (sample_batches_size + threads_per_block - 1) / threads_per_block;
    printf("%d %d %d\n",domain_initial_index, domain_step_size,  domain_log_size);
    // Initialize Structs (TODO: Build Structs in the Bindings)
    M31** new_columns = reinterpret_cast<M31**>(columns);
    CircleDomain domain = CircleDomain(Coset(CirclePointIndex(domain_initial_index), CirclePointIndex(domain_step_size), domain_log_size)); 
    printf("%d %d %d %d\n",domain_size, domain.half_coset.initial_index.idx, domain.half_coset.step_size.idx,  domain.half_coset.log_size);
    printColumnSampleBatchKernel<<<1, 1>>>(sample_batches, domain);

    // Quotient Constants 
    QM31* random_coeffs; 
    cudaMalloc((void**)&random_coeffs, sample_batches_size * sizeof(QM31));
    batch_random_coeffs<<<blocks_per_grid, threads_per_block>>>(random_coeffs, sample_batches, sample_batches_size, random_coeff); 
    QM31** d_denominator_inverses = denominator_inverses(sample_batches, sample_batches_size, domain, domain_size); // Device pointer to device pointers
    
    //printqm31<<<1,1>>>(random_coeffs, sample_batches_size); 

    int threadsPerBlock = 1024;  
    int blocksPerGrid = (domain_size + threadsPerBlock - 1) / threadsPerBlock;
    accumulate_quotients_helper<<<blocksPerGrid, threadsPerBlock>>>(
        values, 
        sample_batches,
        sample_batches_size, 
        domain,
        domain_size,
        columns, 
        columns_size,
        columns_row_size,
        d_denominator_inverses,
        random_coeffs
    );

    //print_values<<<1, 1>>>(values, domain_size);
    print_values_test<<<1, 1>>>(values, domain_size);

    //cudaFree(random_coeffs); 
    // TODO: Properly Free by copying d->h
    // for(size_t i = 0; i < sample_batches_size; ++i) {
    //     cudaFree(d_denominator_inverses[i]);
    // }
    // cudaFree(d_denominator_inverses); 

    //uint32_t* h_values = (uint32_t*)malloc(EXTENSION_DEGREE * domain_size * sizeof(uint32_t)); 
    cudaMemcpy(value_columns1, values, domain_size * sizeof(uint32_t), cudaMemcpyDeviceToHost); 
    cudaMemcpy(value_columns2, values + domain_size, domain_size * sizeof(uint32_t), cudaMemcpyDeviceToHost); 
    cudaMemcpy(value_columns3, values + domain_size * 2, domain_size * sizeof(uint32_t), cudaMemcpyDeviceToHost); 
    cudaMemcpy(value_columns4, values + domain_size * 3, domain_size * sizeof(uint32_t), cudaMemcpyDeviceToHost); 
}

__device__
size_t bit_reverse_index(size_t i, uint32_t log_size) {
    if (log_size == 0) {
        return i;
    }
    #if SIZE_MAX == UINT_MAX
    // 32-bit system
    unsigned int reversed = __brev((unsigned int)i);
    return (size_t)(reversed >> (32 - log_size));

    #elif SIZE_MAX == ULLONG_MAX
        // 64-bit system
        unsigned int lo = (unsigned int)(i & 0xFFFFFFFF);
        unsigned int hi = (unsigned int)(i >> 32);
        unsigned long long reversed = ((unsigned long long)__brev(lo) << 32) | __brev(hi);
        return (size_t)(reversed >> (64 - log_size));
    #endif
}

// __device__ void set(uint32_t** values, size_t row, QM31 row_value) {
//     values[0][row] = row_value.a.a.f;
//     values[1][row] = row_value.a.b.f;
//     values[2][row] = row_value.b.a.f;
//     values[3][row] = row_value.b.b.f;
// }

// offset is domain size
__device__ void set(uint32_t* values, size_t offset, size_t row, QM31 row_value) {
    values[row] = row_value.a.a.f;
    values[offset + row] = row_value.a.b.f;
    values[2 * offset + row] = row_value.b.a.f;
    values[3 * offset + row] = row_value.b.b.f;
}

__global__
void accumulate_quotients_helper (
    uint32_t* values, 
    ColumnSampleBatch* sample_batches,
    size_t sample_batches_size, 
    CircleDomain domain,
    size_t domain_size,
    uint32_t** columns, 
    size_t columns_size,
    size_t columns_row_size,
    QM31** d_denominator_inverses,
    QM31* random_coeffs
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < domain_size) {
        CirclePoint<M31> point = domain.at(bit_reverse_index(row, domain.log_size())); 
        QM31 row_value = accumulate_row_quotients(values, sample_batches, sample_batches_size, columns, columns_size, columns_row_size, d_denominator_inverses, random_coeffs, row);
        set(values, domain_size, row, row_value);
        //set(values, row, row_value); 
    }
}   

__device__ 
QM31 accumulate_row_quotients(
    uint32_t* values, 
    ColumnSampleBatch* sample_batches,
    size_t sample_batches_size, 
    uint32_t** columns, 
    size_t columns_size,
    size_t columns_row_size,
    QM31** d_denominator_inverses,
    QM31* random_coeffs,
    size_t row
) {
    QM31 row_accumulator = QM31::zero();  
    for(int i = 0; i < sample_batches_size; ++i) {
        QM31 numerator = QM31::zero(); 
        QM31 batch_random_coeffs = random_coeffs[i];
        QM31* denominator_inverse = d_denominator_inverses[i]; 

        for(int j = 0; j < sample_batches[i].size; ++j) {
            size_t column_index = sample_batches[i].columns[j]; 
            QM31 sampled_value = sample_batches[i].values[j];

            uint32_t* column = columns[column_index];
            uint32_t value = column[row]; 
            numerator = numerator + sub_from_m31(M31(value), sampled_value);
        }

        row_accumulator = row_accumulator * batch_random_coeffs + numerator * denominator_inverse[row]; 
    }
    return row_accumulator;
}

__global__ void enumerate_fractions(
    QM31* flat_denominators,
    QM31* numerator_terms,
    ColumnSampleBatch* sample_batches,
    size_t sample_batches_size, 
    CircleDomain domain,
    size_t domain_size, 
    size_t n_fractions
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_fractions) {
        size_t sb_idx = tid / domain_size; 
        size_t row = tid % domain_size;
        CirclePoint<M31> domain_point = domain.at(row);
        printf("row: %d, x: %d, y: %d\n", tid, domain_point.x, domain_point.y); 
        // test
        if(tid == 0) {
            CirclePointIndex x = CirclePointIndex(1479063197);
            CirclePointIndex y = CirclePointIndex(2047483648);
            CirclePointIndex z = x - y;
            // uint32_t zz = (x.idx + (1 << 31) - y.idx) & ((1 << 31) - 1);  
            // CirclePoint<M31> zzz = CirclePoint<M31>(M31(1479063197), M31(2047483648));
            // CirclePoint<M31> zzzz = zzz.mul(2047483648);
            // printf(" X + Y CIRCLEPOINT: %d\n", zzz.x); 
            // printf(" X + Y CIRCLEPOINT: %d\n", zzzz.x); 
            printf("ZERO: %d %d\n", z.to_point().x, z.to_point().y); 

        }
        QM31 num;
        QM31 denom;

        point_vanishing_fraction(sample_batches[sb_idx].point, domain_point, &num, &denom); 

        flat_denominators[tid] = num;
        numerator_terms[tid] = denom; 
    } 
}

__global__ void batch_random_coeffs(
    QM31* res,
    ColumnSampleBatch* sample_batches,
    size_t sample_batches_size, 
    QM31 random_coeff
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < sample_batches_size) {
        res[idx] = pow(random_coeff, (uint64_t)sample_batches[idx].size);
    }
}

QM31** denominator_inverses(
    ColumnSampleBatch* sample_batches,
    size_t sample_batches_size, 
    CircleDomain domain,
    size_t domain_size
) {
    // Initialize Return Pointer
    QM31** h_denominator_inverses;  
    QM31** denominator_inverses;  
    size_t n_fractions = sample_batches_size * domain_size;

    cudaMalloc((void**)&denominator_inverses, sample_batches_size * sizeof(QM31*));
    for(size_t i = 0; i < sample_batches_size; ++i) {
        QM31* temp;
        cudaMalloc((void**)&temp, n_fractions * sizeof(QM31)); 
        cudaMemcpy((void**)&denominator_inverses[i], &temp, sizeof(QM31*), cudaMemcpyHostToDevice);
    }

    QM31* flat_denominators;
    QM31* numerator_terms;
    QM31* flat_denominator_inverses;

    cudaMalloc((void**)&flat_denominators, n_fractions * sizeof(QM31)); 
    cudaMalloc((void**)&numerator_terms, n_fractions * sizeof(QM31)); 
    cudaMalloc((void**)&flat_denominator_inverses, n_fractions * sizeof(QM31)); 

    // Populate fraction
    int threads_per_block = 1024; 
    int blocks_per_grid = (n_fractions + threads_per_block - 1) / threads_per_block;
    enumerate_fractions<<<blocks_per_grid, threads_per_block>>>(flat_denominators, numerator_terms, sample_batches, sample_batches_size, domain, domain_size, n_fractions); 
    cudaDeviceSynchronize();

    printqm31<<<1,1>>>(flat_denominators, n_fractions); 
    // Batch Inverse
    batch_inverse_secure_field((qm31*)flat_denominators, (qm31*)flat_denominator_inverses, n_fractions); 
    //printqm31<<<1,1>>>(flat_denominator_inverses, n_fractions); 

    // Numerator Inverse Mul
    denominator_inverses_mul_numerator<<<blocks_per_grid, threads_per_block>>>(flat_denominator_inverses, numerator_terms, n_fractions); 
    //printqm31<<<1,1>>>(flat_denominator_inverses, n_fractions); 

    // // Bit Inverse and Separate
    int blocks_per_grid_chunked = (sample_batches_size + threads_per_block - 1) / threads_per_block;
    denominator_inverses_bit_reverse<<<blocks_per_grid_chunked, threads_per_block>>>(denominator_inverses, flat_denominator_inverses, domain_size); 
    printqm31<<<1,1>>>(flat_denominator_inverses, n_fractions); 
    
    // Host pointer of device pointers which copies into device pointers of device pointers
    QM31** host_ptr = (QM31**)malloc(sample_batches_size * sizeof(QM31*));
    QM31** device_ptr;
    cudaMalloc((void**)&device_ptr, sample_batches_size * sizeof(QM31*));
    cudaMemcpy(host_ptr, denominator_inverses, sample_batches_size * sizeof(QM31*), cudaMemcpyDeviceToHost); 
    for(size_t i = 0; i < sample_batches_size; ++i) {
        bit_reverse_secure_field((qm31*)host_ptr[i], domain_size); 
    }
    cudaMemcpy(device_ptr, host_ptr, sample_batches_size * sizeof(QM31*), cudaMemcpyHostToDevice); 
    //free(host_ptr); 
    //printqm31<<<1,1>>>(device_ptr[0], domain_size);
    
    // cudaDeviceSynchronize();

    // cudaFree(flat_denominators);
    // cudaFree(numerator_terms);
    // cudaFree(flat_denominator_inverses);

    return denominator_inverses; 
}

__global__ void denominator_inverses_mul_numerator(QM31* flat_denominator_inverses, QM31* numerator_terms, size_t n_fractions) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_fractions) {
        flat_denominator_inverses[tid] = flat_denominator_inverses[tid] * numerator_terms[tid]; 
    }
}

__global__ void denominator_inverses_bit_reverse(QM31** denominator_inverses, QM31* flat_denominator_inverses, size_t domain_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < domain_size) {
        cudaMemcpyAsync(denominator_inverses[tid], flat_denominator_inverses, domain_size * sizeof(QM31), cudaMemcpyDeviceToDevice);
    }
}

__device__ void
point_vanishing_fraction(CirclePoint<QM31> vanish_point, CirclePoint<M31> p, QM31* num, QM31* denom) {
    CirclePoint<QM31> p_ef = CirclePoint<QM31>(QM31(CM31(p.x, M31()),CM31()), QM31(CM31(p.y, M31()),CM31()));
    CirclePoint<QM31> h = circle_point_sub(p_ef, vanish_point);
    *num = h.y;
    *denom = QM31::one() + h.x;
}

__global__ void printColumnSampleBatchKernel(const ColumnSampleBatch* batch, CircleDomain domain) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("ColumnSampleBatch contents:\n");
        printf("Point: (%d, %d)\n", batch->point.x, batch->point.y);
        printf("Size: %d\n", batch->size);
        printf("Columns (first 5 or less):");
        for (size_t i = 0; i < batch->size && i < 2; ++i) {
            printf("%d", batch->columns[i]);
        }
        printf("\n");
        printf("Values (first 5 or less):");
        for (size_t i = 0; i < batch->size && i < 2; ++i) {
            printf(" %d", batch->values[i].a.b.f); // Assuming QM31 can be cast to float
        }
        printf("\ndomain: init %d step %d log %d\n", domain.half_coset.initial_index.idx, domain.half_coset.step_size.idx, domain.half_coset.log_size);
        
        printf("\n");
    }
}

__global__ void printqm31(const QM31* batch, size_t size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for(int i = 0; i < size; i++) {
            printf("%d ", batch[i].a.a.f);
            printf("%d ", batch[i].a.b.f);
            printf("%d ", batch[i].b.a.f);
            printf("%d \n", batch[i].b.b.f);
        }
        printf("\n\n");
    }
}

__global__ void print_values(uint32_t** values, size_t domain_size) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            for(int i = 0; i < 4; i++) {
                for(int j = 0; j < domain_size; j++) {
                    printf("M31(%d) ", values[i][j]);
                }
                printf("\n"); 
            }
        }
}

__global__ void print_values_test(uint32_t* values, size_t domain_size) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            for(int i = 0; i < 4 * domain_size; i++) {
                printf("M31(%d) ", values[i]);
            }
            printf("\n"); 
        }
}