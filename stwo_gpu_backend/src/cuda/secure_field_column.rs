use stwo_prover::core::{backend::Column, fields::qm31::SecureField};

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
