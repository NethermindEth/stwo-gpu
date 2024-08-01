use super::bindings::{self, CirclePointSecureField};
use super::secure_field_vec::SecureFieldVec;

use itertools::Itertools;
use stwo_prover::core::fields::qm31::QM31;
use stwo_prover::core::{pcs::quotients};

#[repr(C)]
pub(crate) struct ColumnSampleBatch {
    point: CirclePointSecureField,
    columns: *const usize,
    values: *const u32, // Vec<QM31>
    size: usize 
}

pub struct ColumnSampleBatchVec {
    pub(crate) device_ptr: *const ColumnSampleBatch,
    pub(crate) size: usize
}

impl ColumnSampleBatchVec {
    pub fn from(host_array: &[quotients::ColumnSampleBatch]) -> Self{
        let vec_len = host_array.len(); 
        
        // cudaMalloc device array
        let device_ptr = unsafe {bindings::cuda_malloc_column_sample_batch(vec_len)};

        // Copy host data to device
        for (i, csb) in host_array.iter().enumerate() {
            let (c, v): (Vec<usize>, Vec<QM31>) = csb.columns_and_values.clone().into_iter().unzip(); // TODO:: remove the clone, split into vec<&T>

            let columns = unsafe {bindings::copy_size_t_vec_htd(c.as_ptr(), csb.columns_and_values.len())};
            let value = unsafe {bindings::copy_secure_field_vec_htd(v.as_ptr(), csb.columns_and_values.len())};
            let point = CirclePointSecureField::from(csb.point); 
            unsafe {
                bindings::cuda_set_column_sample_batch(device_ptr, point, columns, value, csb.columns_and_values.len(), i); 
            }
        }

        return Self{device_ptr, size: vec_len}
    }
}