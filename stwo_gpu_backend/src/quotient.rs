use itertools::Itertools;
use stwo_prover::core::{
    fields::{m31::BaseField, qm31::SecureField, secure_column::SecureColumnByCoords},
    pcs::quotients::{ColumnSampleBatch, QuotientOps},
    poly::{
        circle::{CircleDomain, CircleEvaluation, SecureEvaluation},
        BitReversedOrder,
    },
};

use crate::cuda::bindings;
use crate::cuda::bindings::{CirclePointSecureField, CudaSecureField};
use crate::{backend::CudaBackend, cuda::BaseFieldVec};

impl QuotientOps for CudaBackend {
    fn accumulate_quotients(
        domain: CircleDomain,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        sample_batches: &[ColumnSampleBatch],
        _log_blowup_factor: u32,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        let domain_size = domain.size();
        let number_of_columns = columns.len();

        let result: SecureEvaluation<Self, BitReversedOrder> = SecureEvaluation::new(
            domain,
            SecureColumnByCoords {
                columns: [
                    BaseFieldVec::new_uninitialized(domain_size),
                    BaseFieldVec::new_uninitialized(domain_size),
                    BaseFieldVec::new_uninitialized(domain_size),
                    BaseFieldVec::new_uninitialized(domain_size),
                ],
            },
        );

        let device_column_pointers_vector = columns
            .iter()
            .map(|column| column.values.device_ptr)
            .collect_vec();

        unsafe {
            let half_coset_initial_index = domain.half_coset.initial_index;
            let half_coset_step_size = domain.half_coset.step_size;

            let device_column_pointers: *const *const u32 =
                bindings::copy_device_pointer_vec_from_host_to_device(
                    device_column_pointers_vector.as_ptr(),
                    number_of_columns,
                );

            let sample_points: Vec<CirclePointSecureField> = sample_batches
                .iter()
                .map(|column_sample_batch| column_sample_batch.point.into())
                .collect();

            let sample_column_indexes: Vec<u32> = sample_batches
                .iter()
                .flat_map(|column_sample_batch| {
                    column_sample_batch
                        .columns_and_values
                        .iter()
                        .map(|(column, _)| *column as u32)
                        .collect_vec()
                })
                .collect_vec();

            let sample_column_and_values_sizes: Vec<u32> = sample_batches
                .iter()
                .map(|column_sample_batch| column_sample_batch.columns_and_values.len() as u32)
                .collect_vec();

            let sample_column_values: Vec<CudaSecureField> = sample_batches
                .iter()
                .flat_map(|column_sample_batch| {
                    column_sample_batch
                        .columns_and_values
                        .iter()
                        .map(|(_, value)| (*value).into())
                        .collect_vec()
                })
                .collect_vec();

            let flattened_line_coeffs_size =
                number_of_columns * sample_column_and_values_sizes.len() * 3;

            bindings::accumulate_quotients(
                half_coset_initial_index.0 as u32,
                half_coset_step_size.0 as u32,
                domain_size as u32,
                device_column_pointers,
                number_of_columns,
                random_coeff.into(),
                sample_points.as_ptr() as *const u32,
                sample_column_indexes.as_ptr(),
                sample_column_indexes.len() as u32,
                sample_column_values.as_ptr(),
                sample_column_and_values_sizes.as_ptr(),
                sample_points.len() as u32,
                result.values.columns[0].device_ptr,
                result.values.columns[1].device_ptr,
                result.values.columns[2].device_ptr,
                result.values.columns[3].device_ptr,
                flattened_line_coeffs_size as u32,
            );
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use stwo_prover::core::backend::simd::column::BaseColumn;
    use stwo_prover::core::backend::{Column, CpuBackend};
    use stwo_prover::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use stwo_prover::core::fields::m31::{BaseField, M31};
    use stwo_prover::core::fields::qm31::QM31;
    use stwo_prover::core::pcs::quotients::{ColumnSampleBatch, QuotientOps};
    use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation};
    use stwo_prover::core::poly::BitReversedOrder;

    use crate::cuda::BaseFieldVec;
    use crate::CudaBackend;

    #[test]
    fn test_accumulate_quotients_compared_with_cpu() {
        const LOG_SIZE: u32 = 15;
        const LOG_BLOWUP_FACTOR: u32 = 1;
        let small_domain = CanonicCoset::new(LOG_SIZE).circle_domain();
        let domain = CanonicCoset::new(LOG_SIZE + LOG_BLOWUP_FACTOR).circle_domain();
        let e0: BaseColumn = (0..small_domain.size()).map(BaseField::from).collect();
        let e1: BaseColumn = (0..small_domain.size())
            .map(|i| BaseField::from(2 * i))
            .collect();
        let polys = vec![
            CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(
                small_domain,
                BaseFieldVec::from_vec(e0.to_cpu()),
            )
            .interpolate(),
            CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(
                small_domain,
                BaseFieldVec::from_vec(e1.to_cpu()),
            )
            .interpolate(),
        ];
        let columns = vec![polys[0].evaluate(domain), polys[1].evaluate(domain)];
        let random_coeff = QM31::from_m31(M31::from(1), M31::from(2), M31::from(3), M31::from(4));
        let a = polys[0].eval_at_point(SECURE_FIELD_CIRCLE_GEN);
        let b = polys[1].eval_at_point(SECURE_FIELD_CIRCLE_GEN);
        let samples = vec![
            ColumnSampleBatch {
                point: SECURE_FIELD_CIRCLE_GEN,
                columns_and_values: vec![(0, a), (1, b)],
            },
            ColumnSampleBatch {
                point: SECURE_FIELD_CIRCLE_GEN,
                columns_and_values: vec![(0, a), (1, b)],
            },
            ColumnSampleBatch {
                point: SECURE_FIELD_CIRCLE_GEN,
                columns_and_values: vec![(0, a), (1, b)],
            },
        ];
        let cpu_columns = columns
            .iter()
            .map(|c| {
                CircleEvaluation::<CpuBackend, _, BitReversedOrder>::new(
                    c.domain,
                    c.values.to_cpu(),
                )
            })
            .collect_vec();

        let cpu_result = CpuBackend::accumulate_quotients(
            domain,
            &cpu_columns.iter().collect_vec(),
            random_coeff,
            &samples,
            LOG_BLOWUP_FACTOR,
        )
        .values
        .to_vec();

        let gpu_result = CudaBackend::accumulate_quotients(
            domain,
            &columns.iter().collect_vec(),
            random_coeff,
            &samples,
            LOG_BLOWUP_FACTOR,
        )
        .values
        .to_cpu()
        .to_vec();

        assert_eq!(gpu_result, cpu_result);
    }
}
