use stwo_prover::core::{air::accumulation::AccumulationOps, fields::secure_column::SecureColumn};

use crate::backend::CudaBackend;

impl AccumulationOps for CudaBackend {
    fn accumulate(_column: &mut SecureColumn<Self>, _other: &SecureColumn<Self>) {
        todo!()
    }
}
