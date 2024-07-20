use stwo_prover::core::{backend::Column, fields::qm31::SecureField};

#[derive(Clone, Debug)]
pub struct SecureFieldVec{
    pub(crate) device_ptr: *const u32,
    pub(crate) size: usize,
}
