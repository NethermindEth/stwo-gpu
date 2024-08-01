use stwo_prover::core::{
    circle::CirclePoint,
    fields::{m31::BaseField, qm31::SecureField},
};
use stwo_prover::core::vcs::blake2_hash::Blake2sHash;

#[repr(C)]
pub struct CudaSecureField {
    a: BaseField,
    b: BaseField,
    c: BaseField,
    d: BaseField,
}

impl CudaSecureField {
    pub fn zero() -> Self {
        Self {
            a: BaseField::from(0),
            b: BaseField::from(0),
            c: BaseField::from(0),
            d: BaseField::from(0),
        }
    }
}

impl From<SecureField> for CudaSecureField {
    fn from(value: SecureField) -> Self {
        Self {
            a: value.0.0,
            b: value.0.1,
            c: value.1.0,
            d: value.1.1,
        }
    }
}

impl Into<SecureField> for CudaSecureField {
    fn into(self) -> SecureField {
        SecureField::from_m31(self.a, self.b, self.c, self.d)
    }
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

#[repr(C)]
pub(crate) struct CirclePointSecureField {
    x: CudaSecureField,
    y: CudaSecureField,
}

impl From<CirclePoint<SecureField>> for CirclePointSecureField {
    fn from(value: CirclePoint<SecureField>) -> Self {
        Self {
            x: CudaSecureField::from(value.x),
            y: CudaSecureField::from(value.y),
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
        point_x: CudaSecureField,
        point_y: CudaSecureField,
    ) -> CudaSecureField;

    pub fn fold_line(
        gpu_domain: *const u32,
        twiddle_offset: usize,
        n: usize,
        eval_values: *const*const u32,
        alpha: CudaSecureField,
        folded_values: *const*const u32,
    );

    pub fn fold_circle_into_line(
        gpu_domain: *const u32,
        twiddle_offset: usize,
        n: usize,
        eval_values: *const*const u32,
        alpha: CudaSecureField,
        folded_values: *const*const u32,
    );

    pub fn decompose(
        columns: *const*const u32,
        column_size: u32,
        lambda: &CudaSecureField,
        g_values: *const*const u32,
    );

    pub fn accumulate(
        size: u32,
        left_columns: *const *const u32,
        right_columns: *const *const u32,
    );

    pub fn commit_on_first_layer(
        size: usize,
        amount_of_columns: usize,
        columns: *const *const u32,
        result: *mut Blake2sHash,
    );

    pub fn commit_on_layer_with_previous(
        size: usize,
        amount_of_columns: usize,
        columns: *const *const u32,
        previous_layer: *const Blake2sHash,
        result: *mut Blake2sHash,
    );

    pub fn copy_blake_2s_hash_vec_from_host_to_device(
        from: *const Blake2sHash,
        size: usize,
    ) -> *mut Blake2sHash;

    pub fn copy_blake_2s_hash_vec_from_device_to_host(
        from: *const Blake2sHash,
        to: *const Blake2sHash,
        size: usize,
    );

    pub fn free_blake_2s_hash_vec(
        device_pointer: *const Blake2sHash,
    );

    pub fn copy_device_pointer_vec_from_host_to_device(
        from: *const *const u32,
        size: usize,
    ) -> *const *const u32;

    pub fn free_device_pointer_vec(
        device_pointer: *const *const u32,
    );

    pub fn accumulate_quotients(
        domain_initial_point: CirclePointBaseField,
        domain_step: CirclePointBaseField,
        domain_size: u32,
        columns: *const *const u32,
        number_of_columns: usize,
        random_coeff: CudaSecureField,
        sample_points: *const u32,
        sample_columns: *const u32,
        sample_column_and_values_sizes: *const u32,
        result_column_0: *const u32,
        result_column_1: *const u32,
        result_column_2: *const u32,
        result_column_3: *const u32,
    );
}
