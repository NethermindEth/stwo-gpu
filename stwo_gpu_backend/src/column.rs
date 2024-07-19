use stwo_prover::core::{
    backend::{Column, ColumnOps},
    fields::{m31::BaseField, qm31::SecureField},
};

use crate::{
    backend::CudaBackend,
    cuda::{BaseFieldCudaColumn, SecureFieldCudaColumn},
};

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
