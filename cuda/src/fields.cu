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


const uint32_t MODULUS = (1 << 31) - 1; 

__device__ M31 M31::zero() {
    return M31(); 
}

__device__ M31 M31::one() {
    return M31(1); 
}

__device__  M31 M31::operator+(const M31& rhs) const {
    uint32_t out = f + rhs.f; 
    return M31((uint32_t)min(out, out - MODULUS));
}

__device__  M31 M31::operator-(const M31& rhs) const {
    uint32_t out = f - rhs.f;
    return M31((uint32_t)min(out, out + MODULUS));
}

__device__  M31 M31::operator-() const {
    return M31(MODULUS - f);
}

__device__  M31 M31::operator*(const M31& rhs) const {
    unsigned long long int a_e, b_e, prod_e;
    uint32_t prod_lows, prod_highs;

    a_e = (unsigned long long int) f;
    b_e = (unsigned long long int) rhs.f;

    prod_e = a_e * b_e;
    prod_lows = (unsigned long long int) prod_e & 0x7FFFFFFF;
    prod_highs = (unsigned long long int) prod_e >> 31;

    uint32_t out = prod_lows + prod_highs; 
    return M31((uint32_t) min(out, out - MODULUS));
    // uint64_t v = ((uint64_t) f * (uint64_t) rhs.f);
    // uint64_t w = v + (v >> 31);
    // uint64_t u = v + (w >> 31);
    // return M31(u & P);
}

__device__ CM31::CM31() : a(0), b(0) {}
__device__ CM31::CM31(M31 a, M31 b) : a(a), b(b) {}
__device__ CM31::CM31(uint32_t a, uint32_t b) : a(M31(a)), b(M31(b)) {} 

__device__ CM31 CM31::zero() {
    return CM31(); 
}

__device__ CM31 CM31::one() {
    return CM31(M31::one(), M31::zero()); 
}
__device__ CM31 CM31::operator*(const CM31& rhs) const {
    return CM31(
        a * rhs.a - b * rhs.b,
        a * rhs.b + b * rhs.a 
    );
}

__device__ CM31 CM31::operator+(const CM31& rhs) const {
    return CM31(a + rhs.a, b + rhs.b);
}

__device__ CM31 CM31::operator-(const CM31& rhs) const {
    return CM31(a - rhs.a, b - rhs.b);
}

__device__ CM31 CM31::operator-() const {
    return CM31(-a, -b);
}

__device__ CM31 CM31::operator+(const M31& rhs) const {
    return CM31(a + rhs, b); 
}

__device__ QM31::QM31() : a(CM31()), b(CM31()) {}
__device__ QM31::QM31(CM31 a, CM31 b) : a(a), b(b) {}

__device__ QM31 QM31::zero() {
    return QM31();
}

__device__ QM31 QM31::one() {
    return QM31(CM31::one(), CM31::zero()); 
}

__device__ QM31 QM31::operator+(const M31& rhs) const {
    return QM31(a + rhs, b); 
}

__device__ QM31 QM31::operator*(const QM31& rhs) const {
    return QM31(
        a * rhs.a + CM31(M31(2), M31(1)) * b * rhs.b,
        a * rhs.b + b * rhs.a
    );
}

 __device__ QM31 QM31::operator-() const {
     return QM31(-a, -b);
 }

 __device__ QM31 QM31::operator+(const QM31& rhs) const {
     return QM31(a + rhs.a, b + rhs.b);
 }

 __device__ QM31 QM31::operator-(const QM31& rhs) const {
     return QM31(a - rhs.a, b - rhs.b);
 }

 __device__ QM31 square(const QM31& self) {
    return self * self;
 }

// u128?
 __device__ QM31 pow(const QM31& self, uint64_t exp) {
    QM31 res = QM31::one();
    QM31 base = self;
    while (exp > 0) {
            if (exp & 1) {
                res = res * base;
            }
            base = square(base);
            exp >>= 1;
        }
    return res; 
 }

// Acc row quotients helper
 __device__ QM31 sub_from_m31(const M31& lhs, const QM31& rhs) {
     return -rhs + lhs;
 }
