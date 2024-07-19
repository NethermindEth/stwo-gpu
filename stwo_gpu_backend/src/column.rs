use stwo_prover::core::{
    backend::{Column, ColumnOps},
    fields::{m31::BaseField, qm31::SecureField},
};

use crate::backend::CudaBackend;
use crate::cuda;

#[derive(Clone, Debug)]
pub struct BaseFieldCudaColumn {
    pub(crate) device_ptr: *const u32,
    pub(crate) size: usize,
}

impl ColumnOps<BaseField> for CudaBackend {
    type Column = BaseFieldCudaColumn;

    fn bit_reverse_column(column: &mut Self::Column) {
        todo!()
    }
}

impl ColumnOps<SecureField> for CudaBackend {
    type Column = SecureFieldCudaColumn;

    fn bit_reverse_column(column: &mut Self::Column) {
        todo!()
    }
}

impl Column<BaseField> for BaseFieldCudaColumn {
    fn zeros(len: usize) -> Self {
        todo!()
    }

    fn to_cpu(&self) -> Vec<BaseField> {
        cuda::base_field_cuda_column_to_vec(&self)
    }

    fn len(&self) -> usize {
        todo!()
    }

    fn at(&self, index: usize) -> BaseField {
        todo!()
    }

    fn set(&mut self, index: usize, value: BaseField) {
        todo!()
    }
}

impl FromIterator<BaseField> for BaseFieldCudaColumn {
    fn from_iter<T: IntoIterator<Item = BaseField>>(iter: T) -> Self {
        todo!()
    }
}

#[derive(Clone, Debug)]
pub struct SecureFieldCudaColumn(*const u32);

impl Column<SecureField> for SecureFieldCudaColumn {
    fn zeros(len: usize) -> Self {
        todo!()
    }

    fn to_cpu(&self) -> Vec<SecureField> {
        todo!()
    }

    fn len(&self) -> usize {
        todo!()
    }

    fn at(&self, index: usize) -> SecureField {
        todo!()
    }

    fn set(&mut self, index: usize, value: SecureField) {
        todo!()
    }
}

impl FromIterator<SecureField> for SecureFieldCudaColumn {
    fn from_iter<T: IntoIterator<Item = SecureField>>(iter: T) -> Self {
        todo!()
    }
}
