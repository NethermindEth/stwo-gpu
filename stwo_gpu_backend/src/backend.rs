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

impl ColumnOps<Blake2sHash> for CudaBackend {
    type Column = Vec<Blake2sHash>;
    // TODO: implement
    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}

impl MerkleOps<Blake2sMerkleHasher> for CudaBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Vec<Blake2sHash>>,
        columns: &[&BaseFieldVec],
    ) -> Vec<Blake2sHash> {
        // TODO: implement in CUDA
        let columns = columns.iter().map(|col| col.to_vec()).collect_vec();
        // println!("commit on layer input: {:?}, {:?}, {:?}", log_size, prev_layer.unwrap_or(&vec![Blake2sHash::from(vec![0u8; 32])])[0], &columns.iter().collect_vec());
        let res = <CpuBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
            log_size,
            prev_layer,
            &columns.iter().collect_vec(),
        );
        // println!("commit on layer res: {:?}", res[0]);
        res
    }
}
