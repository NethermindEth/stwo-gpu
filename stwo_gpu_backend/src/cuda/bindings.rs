use stwo_prover::core::{
    circle::CirclePoint,
    fields::{m31::BaseField, qm31::SecureField},
};
use stwo_prover::core::fields::m31::M31;

#[link(name = "gpubackend")]
extern "C" {
    pub fn copy_uint32_t_vec_from_device_to_host(
        device_ptr: *const u32,
        host_ptr: *const u32,
        size: u32,
    );
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn copy_uint32_t_vec_from_host_to_device(host_ptr: *const u32, size: u32) -> *const u32;
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn copy_uint32_t_vec_from_device_to_device(
        from: *const u32,
        dst: *const u32,
        size: u32,
    ) -> *const u32;
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn cuda_malloc_uint32_t(size: u32) -> *const u32;
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn cuda_alloc_zeroes_uint32_t(size: u32) -> *const u32;
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn free_uint32_t_vec(device_ptr: *const u32);
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn bit_reverse_base_field(array: *const u32, size: usize);
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn bit_reverse_secure_field(array: *const u32, size: usize);
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn batch_inverse_base_field(from: *const u32, dst: *const u32, size: usize);
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn batch_inverse_secure_field(from: *const u32, dst: *const u32, size: usize);
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn sort_values_and_permute_with_bit_reverse_order(
        from: *const u32,
        size: usize,
    ) -> *const u32;
}

// This is needed since `CirclePoint<BaseField>` is not FFI safe.
#[repr(C)]
pub(crate) struct CirclePointBaseField {
    x: BaseField,
    y: BaseField,
}

impl From<CirclePoint<BaseField>> for CirclePointBaseField {
    fn from(value: CirclePoint<BaseField>) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn precompute_twiddles(
        initial: CirclePointBaseField,
        step: CirclePointBaseField,
        total_size: usize,
    ) -> *const u32;
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn interpolate(values: *const u32, inverse_twiddles_tree: *const u32, values_size: u32);
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn evaluate(values: *const u32, inverse_twiddles_tree: *const u32, values_size: u32);
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn eval_at_point(
        coeffs: *const u32,
        coeffs_size: u32,
        point_x: SecureField,
        point_y: SecureField,
    ) -> SecureField;
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn sum(
        list: *const u32,
        list_size: u32,
    ) -> BaseField;
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn compute_g_values(
        f_values: *const u32,
        size: usize,
        lambda: M31,
    ) -> *const u32;
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn fold_circle(
        gpu_domain: *const u32,
        twiddle_offset: usize,
        n: usize,
        eval_values_1: *const u32,
        eval_values_2: *const u32,
        eval_values_3: *const u32,
        eval_values_4: *const u32,
        alpha: SecureField,
        folded_values_1: *const u32,
        folded_values_2: *const u32,
        folded_values_3: *const u32,
        folded_values_4: *const u32);
}