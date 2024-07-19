use stwo_prover::core::fields::{m31::BaseField, qm31::SecureField, FieldOps};

use crate::backend::CudaBackend;

impl FieldOps<BaseField> for CudaBackend {
    fn batch_inverse(column: &Self::Column, dst: &mut Self::Column) {
        todo!()
    }
}

impl FieldOps<SecureField> for CudaBackend {
    fn batch_inverse(column: &Self::Column, dst: &mut Self::Column) {
        todo!()
    }
}
