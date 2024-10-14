use serde::{Deserialize, Serialize};
use stwo_prover::core::{
    backend::{simd::SimdBackend, Backend, BackendForChannel},
    channel::Blake2sChannel,
    proof_of_work::GrindOps,
    vcs::blake2_merkle::Blake2sMerkleChannel,
};

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct CudaBackend;

impl Backend for CudaBackend {}

impl GrindOps<Blake2sChannel> for CudaBackend {
    fn grind(channel: &Blake2sChannel, pow_bits: u32) -> u64 {
        SimdBackend::grind(channel, pow_bits)
    }
}

impl BackendForChannel<Blake2sMerkleChannel> for CudaBackend {}
