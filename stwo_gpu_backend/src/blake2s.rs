use itertools::Itertools;
use stwo_prover::core::backend::{Column, ColumnOps};
use stwo_prover::core::vcs::blake2_hash::Blake2sHash;
use stwo_prover::core::vcs::blake2_merkle::Blake2sMerkleHasher;
use stwo_prover::core::vcs::ops::{MerkleHasher, MerkleOps};

use crate::cuda::BaseFieldVec;
use crate::CudaBackend;

impl ColumnOps<Blake2sHash> for CudaBackend {
    type Column = Vec<Blake2sHash>;

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
        (0..(1 << log_size))
            .map(|i| {
                Blake2sMerkleHasher::hash_node(
                    prev_layer.map(|prev_layer| (prev_layer[2 * i], prev_layer[2 * i + 1])),
                    &columns.iter().map(|column| column.to_cpu()[i]).collect_vec(),
                )
            })
            .collect()
    }
}
