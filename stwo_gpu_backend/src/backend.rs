use serde::{Deserialize, Serialize};
use stwo_prover::core::backend::{Backend, BackendForChannel};
use stwo_prover::core::channel::Blake2sChannel;
use stwo_prover::core::proof_of_work::GrindOps;
use stwo_prover::core::vcs::blake2_merkle::Blake2sMerkleChannel;

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct CudaBackend;

impl Backend for CudaBackend {}
impl BackendForChannel<Blake2sMerkleChannel> for CudaBackend {}

impl GrindOps<Blake2sChannel> for CudaBackend {
    fn grind(_channel: &Blake2sChannel, _pow_bits: u32) -> u64 {
        todo!()
    }
}