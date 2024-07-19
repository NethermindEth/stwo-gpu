use stwo_prover::core::backend::Backend;

#[derive(Copy, Clone, Debug)]
pub struct CudaBackend;

impl Backend for CudaBackend {}
