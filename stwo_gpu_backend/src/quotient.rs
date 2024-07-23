use stwo_prover::core::{
    fields::{m31::BaseField, qm31::SecureField},
    pcs::quotients::{ColumnSampleBatch, QuotientOps},
    poly::{
        circle::{CircleDomain, CircleEvaluation, SecureEvaluation},
        BitReversedOrder,
    },
};

use crate::backend::CudaBackend;

impl QuotientOps for CudaBackend {
    fn accumulate_quotients(
        _domain: CircleDomain,
        _columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        _random_coeff: SecureField,
        _sample_batches: &[ColumnSampleBatch],
    ) -> SecureEvaluation<Self> {
        todo!()
    }
}
