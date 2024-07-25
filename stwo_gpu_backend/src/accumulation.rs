use stwo_prover::core::{air::accumulation::AccumulationOps, fields::secure_column::SecureColumnByCoords};

use crate::backend::CudaBackend;

impl AccumulationOps for CudaBackend {
    fn accumulate(_column: &mut SecureColumnByCoords<Self>, _other: &SecureColumnByCoords<Self>) {
        todo!()
    }
}
