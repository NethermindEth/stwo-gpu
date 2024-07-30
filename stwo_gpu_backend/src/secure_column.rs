use stwo_prover::core::fields::secure_column::SecureColumnByCoords;

use crate::cuda::BaseFieldVec;
use crate::CudaBackend;

pub struct CudaSecureColumn {
    columns: [*const u32; 4],
}

impl CudaSecureColumn {
    pub unsafe fn new_with_size(size: usize) -> SecureColumnByCoords<CudaBackend> {
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

    pub fn from_secure_column(secure_column: &SecureColumnByCoords<CudaBackend>) -> Self {
        let columns = &secure_column.columns;
        let columns_ptrs_as_vec = columns
            .iter()
            .map(|column| column.device_ptr)
            .collect::<Vec<*const u32>>();
        let columns_ptrs_as_array: [*const u32; 4] = columns_ptrs_as_vec.try_into().unwrap();

        Self {
            columns: columns_ptrs_as_array
        }
    }

    pub fn device_ptr(&self) -> *const*const u32 {
        self.columns.as_ptr()
    }
}
