#ifndef FIELDS_H
#define FIELDS_H

typedef unsigned int uint32_t;
typedef uint32_t m31;
typedef unsigned long long uint64_t;

typedef struct {
    m31 a;
    m31 b;
} cm31;

typedef struct {
    cm31 a;
    cm31 b;
} qm31;

const m31 P = 2147483647;
const cm31 R = {2, 1};

/*##### M31 ##### */

__host__ __device__ m31 mul(m31 a, m31 b);

__host__ __device__ m31 add(m31 a, m31 b);

__host__ __device__ m31 sub(m31 a, m31 b);

__host__ __device__ m31 neg(m31 a);

__host__ __device__ uint64_t pow_to_power_of_two(int n, m31 t);

__host__ __device__ m31 inv(m31 t);

/*##### CM31 ##### */

__host__ __device__ cm31 mul(cm31 x, cm31 y);

__host__ __device__ cm31 add(cm31 x, cm31 y);

__host__ __device__ cm31 sub(cm31 x, cm31 y);

__host__ __device__ cm31 neg(cm31 x);

__host__ __device__ cm31 inv(cm31 t);

/*##### QM31 ##### */

__host__ __device__ qm31 mul(qm31 x, qm31 y);

__host__ __device__ qm31 add(qm31 x, qm31 y);

__host__ __device__ qm31 sub(qm31 x, qm31 y);

__host__ __device__ qm31 inv(qm31 t);

#endif // FIELDS_H