use stwo_prover::core::fields::m31::M31;
use stwo_prover::core::fields::qm31::QM31;
use stwo_prover::core::poly::circle::CircleDomain;
use stwo_prover::core::{
    circle::CirclePoint,
    fields::{m31::BaseField, qm31::SecureField},
};

use super::column_sample_batch_vec::ColumnSampleBatch;
use super::SecureFieldVec;

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

#[link(name = "gpubackend")]
extern "C" {
    pub fn copy_column_circle_evaluation_htd(from: *const u32, column_size: usize, row_size: usize) -> *const *const u32;
    pub fn copy_column_sample_batch_htd(from: *const ColumnSampleBatch, size: usize) -> *const ColumnSampleBatch;

    pub fn unified_malloc_dbl_ptr_uint32_t(size: usize) -> *const *const u32; 
    pub fn unified_set_dbl_ptr_uint32_t(h_out_ptr: *const *const u32, d_in_ptr: *const u32, idx: usize);
    pub fn copy_secure_field_vec_htd(host_ptr: *const QM31, size: usize) -> *const QM31; 
    pub fn copy_size_t_vec_htd(host_ptr: *const usize, size: usize) -> *const usize; 
    pub fn cuda_set_column_sample_batch(device_ptr: *const ColumnSampleBatch, point: CirclePointSecureField, columns: *const usize, values: *const QM31, size: usize, idx: usize);

    pub fn cuda_set_dbl_ptr_uint32_t(h_out_ptr: *const *const u32, size: usize) -> *const *const u32;     
    pub fn cuda_malloc_column_sample_batch(size: usize) -> *const ColumnSampleBatch; 
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn accumulate_quotients(
        value_columns1: *const  u32,
        value_columns2: *const u32,
        value_columns3: *const u32,
        value_columns4: *const u32,
        domain_initial_index: usize,
        domain_step_size: usize,
        domain_log_size: u32, 
        domain_size: usize,
        columns: *const *const u32, 
        columns_size: usize, 
        columns_row_size: usize, 
        random_coeff: QM31,
        sample_batches: *const ColumnSampleBatch, // TODO: QM31 is "not safe" but is mapped properly, use custom struct [u32;4]
        sample_batches_size: usize
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

#[repr(C)]
pub(crate) struct CirclePointSecureField {
    x: QM31,
    y: QM31,
}

impl From<CirclePoint<SecureField>> for CirclePointSecureField {
    fn from(value: CirclePoint<SecureField>) -> Self {
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
    pub fn interpolate(
        eval_domain_size: u32,
        values: *const u32,
        inverse_twiddles_tree: *const u32,
        inverse_twiddle_tree_size: u32,
        values_size: u32,
    );
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn evaluate(
        eval_domain_size: u32,
        values: *const u32,
        twiddles_tree: *const u32,
        twiddle_tree_size: u32,
        values_size: u32,
    );
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
    pub fn sum(list: *const u32, list_size: u32) -> BaseField;
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn compute_g_values(f_values: *const u32, size: usize, lambda: M31) -> *const u32;
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn fold_line(
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
        folded_values_4: *const u32,
    );
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn fold_circle_into_line(
        gpu_domain: *const u32,
        twiddle_offset: usize,
        n: usize,
        eval_values_0: *const u32,
        eval_values_1: *const u32,
        eval_values_2: *const u32,
        eval_values_3: *const u32,
        alpha: SecureField,
        folded_values_0: *const u32,
        folded_values_1: *const u32,
        folded_values_2: *const u32,
        folded_values_3: *const u32,
    );
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn sum_secure_field(
        column_0: *const u32,
        column_1: *const u32,
        column_2: *const u32,
        column_3: *const u32,
        n: u32,
    ) -> SecureField;
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn decompose(
        column_0: *const u32,
        column_1: *const u32,
        column_2: *const u32,
        column_3: *const u32,
        size: u32,
        lambda: &QM31,
        g_value_0: *const u32,
        g_value_1: *const u32,
        g_value_2: *const u32,
        g_value_3: *const u32,
    );
}

#[link(name = "gpubackend")]
extern "C" {
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
