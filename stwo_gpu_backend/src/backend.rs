use serde::{Deserialize, Serialize};
use stwo_prover::core::backend::Backend;

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct CudaBackend;

impl Backend for CudaBackend {}
