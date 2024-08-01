use stwo_prover::core::fields::secure_column::SecureColumnByCoords;

use crate::cuda::{BaseFieldVec, bindings};
use crate::CudaBackend;

pub struct CudaSecureColumn {
    columns: *const *const u32,
}

impl CudaSecureColumn {
    pub fn new_with_size(size: usize) -> SecureColumnByCoords<CudaBackend> {
        let folded_values_column = BaseFieldVec::new_uninitialized(size);

        SecureColumnByCoords {
            columns: [
                folded_values_column.clone(),
                folded_values_column.clone(),
                folded_values_column.clone(),
                folded_values_column,
            ]
        }
    }

    pub fn device_ptr(&self) -> *const*const u32 {
        self.columns
    }
}

impl<'a> From<&'a SecureColumnByCoords<CudaBackend>> for CudaSecureColumn {
    fn from(secure_column: &'a SecureColumnByCoords<CudaBackend>) -> Self {
        let columns = &secure_column.columns;
        let columns_ptrs_as_vec: Vec<*const u32> = columns
            .iter()
            .map(|column| column.device_ptr)
            .collect();

        let columns_ptr = unsafe {
            bindings::copy_device_pointer_vec_from_host_to_device(columns_ptrs_as_vec.as_ptr(), 4)
        };
        Self {
            columns: columns_ptr
        }
    }
}

impl<'a> From<&'a mut SecureColumnByCoords<CudaBackend>> for CudaSecureColumn {
    fn from(secure_column: &'a mut SecureColumnByCoords<CudaBackend>) -> Self {
        let columns = &secure_column.columns;
        let columns_ptrs_as_vec = columns
            .iter()
            .map(|column| column.device_ptr)
            .collect::<Vec<*const u32>>();
        let columns_ptr = unsafe {
            bindings::copy_device_pointer_vec_from_host_to_device(
                columns_ptrs_as_vec.as_ptr(), 4)
        };

        Self {
            columns: columns_ptr
        }
    }
}