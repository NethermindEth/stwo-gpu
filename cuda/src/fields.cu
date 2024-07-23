#include "../include/fields.cuh"

__host__ __device__ m31 mul(m31 a, m31 b) {
    uint64_t v = ((uint64_t) a * (uint64_t) b);
    uint64_t w = v + (v >> 31);
    uint64_t u = v + (w >> 31);
    return u & P;
}

__host__ __device__ m31 add(m31 a, m31 b) {
    uint64_t sum = ((uint64_t) a + (uint64_t) b);
    return min(sum, sum - P);
}

__host__ __device__ m31 sub(m31 a, m31 b) {
    return add(a, P - b);
}

__host__ __device__ m31 neg(m31 a) {
    return P - a;
}

__host__ __device__ uint64_t pow_to_power_of_two(int n, m31 t) {
    int i = 0;
    while (i < n) {
        t = mul(t, t);
        i++;
    }
    return t;
}

__host__ __device__ m31 inv(m31 t) {
    uint64_t t0 = mul(pow_to_power_of_two(2, t), t);
    uint64_t t1 = mul(pow_to_power_of_two(1, t0), t0);
    uint64_t t2 = mul(pow_to_power_of_two(3, t1), t0);
    uint64_t t3 = mul(pow_to_power_of_two(1, t2), t0);
    uint64_t t4 = mul(pow_to_power_of_two(8, t3), t3);
    uint64_t t5 = mul(pow_to_power_of_two(8, t4), t3);
    return mul(pow_to_power_of_two(7, t5), t2);
}

/*##### CM31 ##### */

__host__ __device__ cm31 mul(cm31 x, cm31 y) {
    return {sub(mul(x.a, y.a), mul(x.b, y.b)), add(mul(x.a, y.b), mul(x.b, y.a))};
}

__host__ __device__ cm31 add(cm31 x, cm31 y) {
    return {add(x.a, y.a), add(x.b, y.b)};
}

__host__ __device__ cm31 sub(cm31 x, cm31 y) {
    return {sub(x.a, y.a), sub(x.b, y.b)};
}

__host__ __device__ cm31 neg(cm31 x) {
    return {neg(x.a), neg(x.b)};
}

__host__ __device__ cm31 inv(cm31 t) {
    m31 factor = inv(add(mul(t.a, t.a), mul(t.b, t.b)));
    return {mul(t.a, factor), mul(neg(t.b), factor)};
}

__host__ __device__ cm31 mul_by_scalar(cm31 x, m31 scalar) {
    return cm31 { mul(x.a, scalar), mul(x.b, scalar) };
}

/*##### QM31 ##### */

__host__ __device__ qm31 mul(qm31 x, qm31 y) {
    // Karatsuba multiplication
    cm31 v0 = mul(x.a, y.a);
    cm31 v1 = mul(x.b, y.b);
    cm31 v2 = mul(add(x.a, x.b), add(y.a, y.b));
    return {
            add(v0, mul(R, v1)),
            sub(v2, add(v0, v1))
    };
}

__host__ __device__ qm31 add(qm31 x, qm31 y) {
    return {add(x.a, y.a), add(x.b, y.b)};
}

__host__ __device__ qm31 sub(qm31 x, qm31 y) {
    return {sub(x.a, y.a), sub(x.b, y.b)};
}

__host__ __device__ qm31 mul_by_scalar(qm31 x, m31 scalar) {
    return qm31 { mul_by_scalar(x.a, scalar), mul_by_scalar(x.b, scalar) };
}

__host__ __device__ qm31 inv(qm31 t) {
    cm31 b2 = mul(t.b, t.b);
    cm31 ib2 = {neg(b2.b), b2.a};
    cm31 denom = sub(mul(t.a, t.a), add(add(b2, b2), ib2));
    cm31 denom_inverse = inv(denom);
    return {mul(t.a, denom_inverse), neg(mul(t.b, denom_inverse))};
}