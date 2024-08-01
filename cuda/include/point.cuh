#ifndef POINT_H
#define POINT_H

#include "fields.cuh"

typedef struct {
    m31 x;
    m31 y;
} point;

typedef struct {
    qm31 x;
    qm31 y;
} secure_field_point;

const point m31_circle_gen = {2, 1268011823};

/*##### Point ##### */

__host__ __device__ __forceinline__ point one() {
    return {1, 0};
}

__host__ __device__ __forceinline__ point point_mul(point &p1, point &p2) {
    return {
        sub(mul(p1.x, p2.x), mul(p1.y, p2.y)),
        add(mul(p1.x, p2.y), mul(p1.y, p2.x)),
    };
}

__host__ __device__ __forceinline__ point point_square(point &p1) {
    return point_mul(p1, p1);
}

__host__ __device__ __forceinline__ point point_pow(point p, int exponent) {
    point result = one();
    while (exponent > 0) {
        if (exponent & 1) {
            result = point_mul(p, result);
        }
        p = point_square(p);
        exponent >>= 1;
    }
    return result;
}

#endif // POINT_H