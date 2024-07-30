use itertools::Itertools;
use stwo_prover::core::{
    backend::CpuBackend,
    fields::{m31::BaseField, qm31::SecureField, secure_column::SecureColumnByCoords},
    pcs::quotients::{ColumnSampleBatch, QuotientOps},
    poly::{
        BitReversedOrder,
        circle::{CircleDomain, CircleEvaluation, SecureEvaluation},
    },
};

use crate::{backend::CudaBackend, cuda::BaseFieldVec};

impl QuotientOps for CudaBackend {
    fn accumulate_quotients(
        domain: CircleDomain,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        sample_batches: &[ColumnSampleBatch],
    ) -> SecureEvaluation<Self> {
        let columns = columns
            .iter()
            .map(|column| {
                CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
                    column.domain,
                    column.to_vec(),
                )
            })
            .collect_vec();
        let cpu_result = <CpuBackend as QuotientOps>::accumulate_quotients(
            domain,
            columns.iter().collect_vec().as_slice(),
            random_coeff,
            sample_batches,
        );
        SecureEvaluation {
            domain: cpu_result.domain,
            values: SecureColumnByCoords {
                columns: [
                    BaseFieldVec::from_vec(cpu_result.values.columns[0].clone()),
                    BaseFieldVec::from_vec(cpu_result.values.columns[1].clone()),
                    BaseFieldVec::from_vec(cpu_result.values.columns[2].clone()),
                    BaseFieldVec::from_vec(cpu_result.values.columns[3].clone()),
                ],
            },
        }
    }
}
