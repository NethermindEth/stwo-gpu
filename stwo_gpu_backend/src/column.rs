use stwo_prover::core::{
    backend::{ColumnOps},
    fields::{m31::BaseField, qm31::SecureField},
};

use crate::{
    backend::CudaBackend,
    cuda
};

impl ColumnOps<BaseField> for CudaBackend {
    type Column = cuda::BaseFieldVec;

    fn bit_reverse_column(column: &mut Self::Column) {
        todo!()
    }
}

impl ColumnOps<SecureField> for CudaBackend {
    type Column = cuda::SecureFieldVec;

    fn bit_reverse_column(column: &mut Self::Column) {
        todo!()
    }
}


impl Column<BaseField> for BaseFieldVec {
    fn zeros(len: usize) -> Self {
        todo!()
    }

    fn to_cpu(&self) -> Vec<BaseField> {
        self.to_vec()
    }

    fn len(&self) -> usize {
        self.size
    }

    fn at(&self, index: usize) -> BaseField {
        todo!()
    }

    fn set(&mut self, index: usize, value: BaseField) {
        todo!()
    }
}

impl FromIterator<BaseField> for BaseFieldVec {
    fn from_iter<T: IntoIterator<Item = BaseField>>(iter: T) -> Self {
        todo!()
    }
}

impl Column<SecureField> for SecureFieldVec {
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

impl FromIterator<SecureField> for SecureFieldVec {
    fn from_iter<T: IntoIterator<Item = SecureField>>(iter: T) -> Self {
        todo!()
    }
}
