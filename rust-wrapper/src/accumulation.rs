use stwo_prover::core::{air::accumulation::AccumulationOps, fields::secure_column::SecureColumn};

use crate::backend::CudaBackend;

impl AccumulationOps for CudaBackend {
    fn accumulate(column: &mut SecureColumn<Self>, other: &SecureColumn<Self>) {
        todo!()
    }
}
