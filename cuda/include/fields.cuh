#ifndef FIELDS_H
#define FIELDS_H
#include <stdint.h>

typedef uint32_t m31;

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

__host__ __device__ cm31 mul_by_scalar(cm31 x, m31 scalar);

/*##### QM31 ##### */

__host__ __device__ qm31 mul(qm31 x, qm31 y);

__host__ __device__ qm31 add(qm31 x, qm31 y);

__host__ __device__ qm31 sub(qm31 x, qm31 y);

__host__ __device__ qm31 inv(qm31 t);

__host__ __device__ qm31 mul_by_scalar(qm31 x, m31 scalar);


struct M31 {
    uint32_t f; // field

    __host__ __device__ M31() : f(0) {}
    __host__ __device__ explicit M31(uint32_t value) : f(value) {}

    __device__ static M31 zero();
    __device__ static M31 one();
    //__device__ constexpr static M31 from_unsigned_int(uint32_t val);

    __device__ M31 operator+(const M31& rhs) const;
    __device__ M31 operator-(const M31& rhs) const;
    __device__ M31 operator-() const;
    __device__ M31 operator*(const M31& rhs) const;
};

struct CM31 {
    M31 a;
    M31 b;

    __host__ __device__ CM31();
    __host__ __device__ CM31(M31 a, M31 b);
    __host__ __device__ CM31(uint32_t a, uint32_t b); 

    __device__ static CM31 zero();
    __device__ static CM31 one();

    __device__ CM31 operator*(const CM31& rhs) const;
    __device__ CM31 operator+(const CM31& rhs) const;
    __device__ CM31 operator-(const CM31& rhs) const;
    __device__ CM31 operator-() const;

    __device__ CM31 operator+(const M31& rhs) const;
};

struct QM31 {
    CM31 a;
    CM31 b;

    __host__ __device__ QM31();
    __host__ __device__ QM31(CM31 a, CM31 b);

    __device__ static QM31 zero();
    __device__ static QM31 one();

    __device__ QM31 operator+(const QM31& rhs) const;
    __device__ QM31 operator-(const QM31& rhs) const;
    __device__ QM31 operator-() const;
    __device__ QM31 operator*(const QM31& rhs) const;

    __device__ QM31 operator+(const M31& rhs) const; 
};
#endif // FIELDS_H