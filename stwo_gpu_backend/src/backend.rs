use itertools::Itertools;
use serde::{Deserialize, Serialize};
use stwo_prover::core::{
    backend::{Backend, ColumnOps, CpuBackend},
    vcs::{
        blake2_hash::Blake2sHash,
        blake2_merkle::Blake2sMerkleHasher,
        ops::MerkleOps,
    },
};

use crate::cuda::BaseFieldVec;

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct CudaBackend;

impl Backend for CudaBackend {}
