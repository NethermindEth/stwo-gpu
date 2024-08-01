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

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use stwo_prover::core::backend::{Column, CpuBackend};
    use stwo_prover::core::backend::simd::column::BaseColumn;
    use stwo_prover::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use stwo_prover::core::fields::m31::{BaseField, M31};
    use stwo_prover::core::fields::qm31::QM31;
    use stwo_prover::core::pcs::quotients::{ColumnSampleBatch, QuotientOps};
    use stwo_prover::core::poly::BitReversedOrder;
    use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation};
    use stwo_prover::core::prover::LOG_BLOWUP_FACTOR;

    use crate::cuda::BaseFieldVec;
    use crate::CudaBackend;

    #[test]
    fn test_accumulate_quotients_compared_with_cpu() {
        const LOG_SIZE: u32 = 8;
        let small_domain = CanonicCoset::new(LOG_SIZE).circle_domain();
        let domain = CanonicCoset::new(LOG_SIZE + LOG_BLOWUP_FACTOR).circle_domain();
        let e0: BaseColumn = (0..small_domain.size()).map(BaseField::from).collect();
        let e1: BaseColumn = (0..small_domain.size())
            .map(|i| BaseField::from(2 * i))
            .collect();
        let polys = vec![
            CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(small_domain, BaseFieldVec::from_vec(e0.to_cpu()))
                .interpolate(),
            CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(small_domain, BaseFieldVec::from_vec(e1.to_cpu()))
                .interpolate(),
        ];
        let columns = vec![polys[0].evaluate(domain), polys[1].evaluate(domain)];
        let random_coeff = QM31::from_m31(M31::from(1), M31::from(2), M31::from(3), M31::from(4));
        let a = polys[0].eval_at_point(SECURE_FIELD_CIRCLE_GEN);
        let b = polys[1].eval_at_point(SECURE_FIELD_CIRCLE_GEN);
        let samples = vec![ColumnSampleBatch {
            point: SECURE_FIELD_CIRCLE_GEN,
            columns_and_values: vec![(0, a), (1, b)],
        }];
        let cpu_columns = columns
            .iter()
            .map(|c| {
                CircleEvaluation::<CpuBackend, _, BitReversedOrder>::new(
                    c.domain,
                    c.values.to_cpu(),
                )
            })
            .collect::<Vec<_>>();
        let cpu_result = CpuBackend::accumulate_quotients(
            domain,
            &cpu_columns.iter().collect_vec(),
            random_coeff,
            &samples,
        ).values.to_vec();

        let gpu_result  = CudaBackend::accumulate_quotients(
            domain,
            &columns.iter().collect_vec(),
            random_coeff,
            &samples,
        ).values.to_cpu().to_vec();

        assert_eq!(gpu_result, cpu_result);
    }
}