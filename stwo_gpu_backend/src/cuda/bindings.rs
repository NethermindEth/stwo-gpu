use stwo_prover::core::fields::qm31::QM31;
use stwo_prover::core::{
    circle::CirclePoint,
    fields::{m31::BaseField, qm31::SecureField},
};
use stwo_prover::core::vcs::blake2_hash::Blake2sHash;

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

    pub fn decompose(
        columns: *const*const u32,
        column_size: u32,
        lambda: &QM31,
        g_values: *const*const u32,
    );

    pub fn accumulate(
        size: u32,
        left_columns: *const *const u32,
        right_columns: *const *const u32,
    );
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn commit_on_first_layer(
        size: usize,
        amount_of_columns: usize,
        columns: *const *const u32,
        result: *mut Blake2sHash,
    );
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn commit_on_layer_with_previous(
        size: usize,
        amount_of_columns: usize,
        columns: *const *const u32,
        previous_layer: *const Blake2sHash,
        result: *mut Blake2sHash,
    );
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn copy_blake_2s_hash_from_host_to_device(
        from: *const Blake2sHash,
    ) -> *mut Blake2sHash;
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn copy_blake_2s_hash_from_device_to_host(
        from: *const Blake2sHash,
        to: *const Blake2sHash,
    );
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn free_blake_2s_hash(
        device_pointer: *const Blake2sHash,
    );
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn copy_blake_2s_hash_vec_from_host_to_device(
        from: *const Blake2sHash,
        size: usize,
    ) -> *mut Blake2sHash;
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn copy_blake_2s_hash_vec_from_device_to_host(
        from: *const Blake2sHash,
        to: *const Blake2sHash,
        size: usize,
    );
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn free_blake_2s_hash_vec(
        device_pointer: *const Blake2sHash,
    );
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn copy_device_pointer_vec_from_host_to_device(
        from: *const *const u32,
        size: usize,
    ) -> *const *const u32;
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn free_device_pointer_vec(
        device_pointer: *const *const u32,
    );
}
