use stwo_prover::core::fields::m31::M31;
use stwo_prover::core::fields::qm31::QM31;
use stwo_prover::core::{
    circle::CirclePoint,
    fields::{m31::BaseField, qm31::SecureField},
};

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
    pub fn copy_uint32_t_vec_from_device_to_host(
        device_ptr: *const u32,
        host_ptr: *const u32,
        size: u32,
    );

    pub fn copy_uint32_t_vec_from_host_to_device(host_ptr: *const u32, size: u32) -> *const u32;

    pub fn copy_uint32_t_vec_from_device_to_device(
        from: *const u32,
        dst: *const u32,
        size: u32,
    ) -> *const u32;

    pub fn cuda_malloc_uint32_t(size: u32) -> *const u32;

    pub fn cuda_alloc_zeroes_uint32_t(size: u32) -> *const u32;

    pub fn free_uint32_t_vec(device_ptr: *const u32);

    pub fn bit_reverse_base_field(array: *const u32, size: usize);

    pub fn bit_reverse_secure_field(array: *const u32, size: usize);

    pub fn batch_inverse_base_field(from: *const u32, dst: *const u32, size: usize);

    pub fn batch_inverse_secure_field(from: *const u32, dst: *const u32, size: usize);

    pub fn sort_values_and_permute_with_bit_reverse_order(
        from: *const u32,
        size: usize,
    ) -> *const u32;

    pub fn precompute_twiddles(
        initial: CirclePointBaseField,
        step: CirclePointBaseField,
        total_size: usize,
    ) -> *const u32;

    pub fn interpolate(
        eval_domain_size: u32,
        values: *const u32,
        inverse_twiddles_tree: *const u32,
        inverse_twiddle_tree_size: u32,
        values_size: u32,
    );

    pub fn evaluate(
        eval_domain_size: u32,
        values: *const u32,
        twiddles_tree: *const u32,
        twiddle_tree_size: u32,
        values_size: u32,
    );

    pub fn eval_at_point(
        coeffs: *const u32,
        coeffs_size: u32,
        point_x: SecureField,
        point_y: SecureField,
    ) -> SecureField;

    pub fn sum(list: *const u32, list_size: u32) -> BaseField;

    pub fn compute_g_values(f_values: *const u32, size: usize, lambda: M31) -> *const u32;

    pub fn fold_line(
        gpu_domain: *const u32,
        twiddle_offset: usize,
        n: usize,
        eval_values: *const*const u32,
        alpha: SecureField,
        folded_values: *const*const u32,
    );

    pub fn fold_circle_into_line(
        gpu_domain: *const u32,
        twiddle_offset: usize,
        n: usize,
        eval_values: *const*const u32,
        alpha: SecureField,
        folded_values: *const*const u32,
    );

    pub fn sum_secure_field(
        column_0: *const u32,
        column_1: *const u32,
        column_2: *const u32,
        column_3: *const u32,
        n: u32,
    ) -> SecureField;

    pub fn decompose(
        columns: *const*const u32,
        column_size: u32,
        lambda: &QM31,
        g_values: *const*const u32,
    );

    pub fn accumulate(
        size: u32,
        left_column_0: *const u32,
        left_column_1: *const u32,
        left_column_2: *const u32,
        left_column_3: *const u32,
        right_column_0: *const u32,
        right_column_1: *const u32,
        right_column_2: *const u32,
        right_column_3: *const u32,
    );
}
