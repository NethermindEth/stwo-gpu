#ifndef CIRCLE_H
#define CIRCLE_H

#include "fields.cuh"
#include "point.cuh"

extern "C"
m31* sort_values_and_permute_with_bit_reverse_order(m31 *from, int size);

extern "C"
m31* precompute_twiddles(point initial, point step, int total_size);

extern "C"
void interpolate(int eval_domain_size, m31 *values, m31 *inverse_twiddles_tree, int inverse_twiddles_size, int values_size);

extern "C"
void evaluate(int eval_domain_size, m31 *values, m31 *twiddles_tree, int twiddles_size, int values_size);

extern "C"
qm31 eval_at_point(m31 *coeffs, int coeffs_size, qm31 point_x, qm31 point_y);

__device__ int get_twiddle(m31 *twiddles, int index);

template<typename F>
struct CirclePoint {
    F x;
    F y;
    
    __host__ __device__ CirclePoint();
    __host__ __device__ CirclePoint(F x, F y);

    __device__ CirclePoint<F> conjugate() const;
    __device__ CirclePoint<F> antipode() const;
    __device__ CirclePoint<F> zero() const;
    __device__ CirclePoint<F> double_val() const;

    __device__  CirclePoint<F> operator+(const CirclePoint<F>& rhs) const;
    __device__  CirclePoint<F> operator-() const;

    __device__ CirclePoint<F> mul(unsigned long long scalar) const;
};

template<typename F>
__device__ 
CirclePoint<F> circle_point_sub(const CirclePoint<F>& lhs, const CirclePoint<F>& rhs);

struct CirclePointIndex {
    uint32_t idx;
    
    __host__ __device__ CirclePointIndex(size_t idx);

    __device__ static CirclePointIndex zero();
    __device__ static CirclePointIndex generator();
    __device__ CirclePointIndex reduce() const;
    __device__ static CirclePointIndex subgroup_gen(unsigned int log_size);
    __device__ CirclePoint<M31> to_point() const;
    __device__ CirclePointIndex half() const;
    
    __device__ CirclePointIndex operator+(const CirclePointIndex& rhs) const;
    __device__ CirclePointIndex operator-(const CirclePointIndex& rhs) const;
    __device__ CirclePointIndex operator*(size_t rhs) const;
    __device__ CirclePointIndex operator-() const;
    
    __device__ CirclePointIndex mul(size_t rhs) const;
};


struct Coset {
    CirclePointIndex initial_index;
    //CirclePoint<M31>  initial;
    CirclePointIndex step_size;
    //CirclePoint<M31>  step;
    uint32_t log_size;

    __host__ __device__ Coset(const CirclePointIndex& initial, const CirclePointIndex& step, uint32_t size);

    __device__ size_t size() const;
    __device__ CirclePointIndex index_at(size_t i) const;
};

struct CircleDomain {
    Coset half_coset;

    __host__ __device__ CircleDomain(const Coset& half_coset); 

    __device__ uint32_t log_size(); 
    __device__ size_t size();
    __device__ CirclePoint<M31> at(size_t i);
    __device__ CirclePointIndex index_at(size_t i); 
};

struct CircleEvaluation {
    CircleDomain domain;
    M31* values; 
};

#endif // CIRCLE_H