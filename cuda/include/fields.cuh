#ifndef FIELDS_H
#define FIELDS_H

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

struct m31 {
    uint32_t value;

    __host__ __device__ operator unsigned int() const {
        return (unsigned int) value;
    }

    __host__ __device__ m31(const uint32_t value) {
        this->value = value;
    }

    __host__ __device__ __forceinline__ static m31 P() {
        return 2147483647;
    }
};

struct cm31{
    m31 a;
    m31 b;

    __host__ __device__ cm31 operator*(const cm31& multiplier) const;
    __host__ __device__ cm31 operator+(const cm31& addend) const;
    __host__ __device__ cm31 operator-(const cm31& subtrahend) const;
    __host__ __device__ cm31 operator*(const m31& subtrahend) const;
    __host__ __device__ cm31 inverse() const;

    __host__ __device__ __forceinline__ static cm31 R() {
        return {2, 1};
    }
};

struct qm31 {
    cm31 a;
    cm31 b;

    __host__ __device__ qm31 operator*(const qm31& multiplier) const;
    __host__ __device__ qm31 operator+(const qm31& addend) const;
    __host__ __device__ qm31 operator-(const qm31& subtrahend) const;
    __host__ __device__ qm31 operator*(const m31& subtrahend) const;
    __host__ __device__ qm31 inverse() const;
};

//const uint32_t P = 2147483647;
//const cm31 R = {2, 1};

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

__host__ __device__ cm31 mul_by_scalar(cm31 x, m31 scalar);

/*##### QM31 ##### */

__host__ __device__ qm31 mul(qm31 x, qm31 y);

__host__ __device__ qm31 add(qm31 x, qm31 y);

__host__ __device__ qm31 sub(qm31 x, qm31 y);

__host__ __device__ qm31 inv(qm31 t);

__host__ __device__ qm31 mul_by_scalar(qm31 x, m31 scalar);

#endif // FIELDS_H