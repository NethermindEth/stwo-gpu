use itertools::Itertools;
use rand::seq::index::sample;
use stwo_prover::core::{
    backend::CpuBackend,
    fields::{m31::BaseField, qm31::SecureField, secure_column::SecureColumnByCoords},
    pcs::quotients::{ColumnSampleBatch, QuotientOps},
    poly::{
        BitReversedOrder,
        circle::{CircleDomain, CircleEvaluation, SecureEvaluation},
    },
};
use stwo_prover::core::circle::CirclePoint;
use stwo_prover::core::constraints::complex_conjugate_line_coeffs;
use stwo_prover::core::fields::cm31::CM31;
use stwo_prover::core::fields::FieldExpOps;
use stwo_prover::core::fields::m31::M31;
use stwo_prover::core::pcs::quotients::PointSample;
use crate::{backend::CudaBackend, cuda::BaseFieldVec};
use crate::cuda::bindings;
use crate::cuda::bindings::{CirclePointBaseField, CirclePointSecureField, CudaSecureField};

impl QuotientOps for CudaBackend {
    fn accumulate_quotients(
        domain: CircleDomain,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        sample_batches: &[ColumnSampleBatch],
    ) -> SecureEvaluation<Self> {
        let domain_size = domain.size();
        let number_of_columns = columns.len();

        let mut result: SecureEvaluation<CudaBackend> = SecureEvaluation {
            domain,
            values: SecureColumnByCoords {
                columns: [
                    BaseFieldVec::new_uninitialized(domain_size),
                    BaseFieldVec::new_uninitialized(domain_size),
                    BaseFieldVec::new_uninitialized(domain_size),
                    BaseFieldVec::new_uninitialized(domain_size),
                ],
            },
        };

        let device_column_pointers_vector = columns.iter()
            .map(|column| column.values.device_ptr)
            .collect_vec();

        let quotient_constants = quotient_constants(sample_batches, random_coeff);

        unsafe {
            let half_coset_initial_index = domain.half_coset.initial_index;
            let half_coset_step_size = domain.half_coset.step_size;

            let device_column_pointers: *const *const u32 = bindings::copy_device_pointer_vec_from_host_to_device(
                device_column_pointers_vector.as_ptr(), number_of_columns
            );

            let sample_points: Vec<CirclePointSecureField> = sample_batches.iter().map(|column_sample_batch|
                column_sample_batch.point.into()
            ).collect();

            let sample_column_indexes: Vec<u32> = sample_batches.iter().flat_map( |column_sample_batch|
                column_sample_batch.columns_and_values.iter().map(|(column, _)|
                    *column as u32
                ).collect_vec()
            ).collect_vec();

            let sample_column_and_values_sizes: Vec<u32> = sample_batches.iter().map( |column_sample_batch|
                column_sample_batch.columns_and_values.len() as u32,
            ).collect_vec();

            let sample_column_values: Vec<CudaSecureField> = sample_batches.iter().flat_map( |column_sample_batch|
                column_sample_batch.columns_and_values.iter().map(|(_, value)|
                    (*value).into()
                ).collect_vec()
            ).collect_vec();

            let line_coeffs_sizes = quotient_constants.line_coeffs.iter().map( |vector| vector.len()).collect_vec();

            let flattened_line_coeffs = quotient_constants.line_coeffs.into_iter().flat_map( |vector: Vec<(SecureField, SecureField, SecureField)>|
                vector.into_iter().flat_map( |(x, y, z)|
                    vec![x, y, z]
                ).collect_vec()
            ).collect_vec();

            bindings::accumulate_quotients(
                half_coset_initial_index.0 as u32,
                half_coset_step_size.0 as u32,
                domain_size as u32,
                device_column_pointers,
                number_of_columns,
                random_coeff.into(),
                sample_points.as_ptr() as *const u32,
                sample_column_indexes.as_ptr(),
                sample_column_values.as_ptr(),
                sample_column_and_values_sizes.as_ptr(),
                sample_points.len() as u32,
                result.values.columns[0].device_ptr,
                result.values.columns[1].device_ptr,
                result.values.columns[2].device_ptr,
                result.values.columns[3].device_ptr,
                flattened_line_coeffs.as_ptr() as *const u32,
                line_coeffs_sizes.as_ptr() as *const u32,
                quotient_constants.batch_random_coeffs.as_ptr() as *const u32
            );
        }

        return result;
    }
}

fn denominator_inverse(
    sample_batches: &[ColumnSampleBatch],
    domain_point: CirclePoint<BaseField>,
) -> Vec<CM31> {
    let mut flat_denominators = Vec::with_capacity(sample_batches.len());
    // We want a P to be on a line that passes through a point Pr + uPi in QM31^2, and its conjugate
    // Pr - uPi. Thus, Pr - P is parallel to Pi. Or, (Pr - P).x * Pi.y - (Pr - P).y * Pi.x = 0.
    for sample_batch in sample_batches {
        // Extract Pr, Pi.
        let prx = sample_batch.point.x.0;
        let pry = sample_batch.point.y.0;
        let pix = sample_batch.point.x.1;
        let piy = sample_batch.point.y.1;
        flat_denominators.push(
            (prx - domain_point.x) * piy - (pry - domain_point.y) * pix);
    }

    let mut flat_denominator_inverses = vec![CM31::default(); flat_denominators.len()];
    CM31::batch_inverse(&flat_denominators, &mut flat_denominator_inverses);

    flat_denominator_inverses
}

/// Holds the precomputed constant values used in each quotient evaluation.
pub struct QuotientConstants {
    /// The line coefficients for each quotient numerator term. For more details see
    /// [self::column_line_coeffs].
    pub line_coeffs: Vec<Vec<(SecureField, SecureField, SecureField)>>,
    /// The random coefficients used to linearly combine the batched quotients For more details see
    /// [self::batch_random_coeffs].
    pub batch_random_coeffs: Vec<SecureField>,
}

pub fn quotient_constants(
    sample_batches: &[ColumnSampleBatch],
    random_coeff: SecureField,
) -> QuotientConstants {
    let line_coeffs = column_line_coeffs(sample_batches, random_coeff);
    let batch_random_coeffs = batch_random_coeffs(sample_batches, random_coeff);
    QuotientConstants {
        line_coeffs,
        batch_random_coeffs,
    }
}

/// Precompute the random coefficients used to linearly combine the batched quotients.
/// Specifically, for each sample batch we compute random_coeff^(number of columns in the batch),
/// which is used to linearly combine the batch with the next one.
pub fn batch_random_coeffs(
    sample_batches: &[ColumnSampleBatch],
    random_coeff: SecureField,
) -> Vec<SecureField> {
    sample_batches
        .iter()
        .map(|sb| random_coeff.pow(sb.columns_and_values.len() as u128))
        .collect()
}

/// Precompute the complex conjugate line coefficients for each column in each sample batch.
/// Specifically, for the i-th (in a sample batch) column's numerator term
/// `alpha^i * (c * F(p) - (a * p.y + b))`, we precompute and return the constants:
/// (`alpha^i * a`, `alpha^i * b`, `alpha^i * c`).
pub fn column_line_coeffs(
    sample_batches: &[ColumnSampleBatch],
    random_coeff: SecureField,
) -> Vec<Vec<(SecureField, SecureField, SecureField)>> {
    sample_batches
        .iter()
        .map(|sample_batch| {
            let mut alpha = SecureField::from_m31(M31::from(1), M31::from(0), M31::from(0), M31::from(0));
            sample_batch
                .columns_and_values
                .iter()
                .map(|(_, sampled_value)| {
                    alpha *= random_coeff;
                    let sample = PointSample {
                        point: sample_batch.point,
                        value: *sampled_value,
                    };
                    complex_conjugate_line_coeffs(&sample, alpha)
                })
                .collect()
        })
        .collect()
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
        const LOG_SIZE: u32 = 3;
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

        let gpu_result = CudaBackend::accumulate_quotients(
            domain,
            &columns.iter().collect_vec(),
            random_coeff,
            &samples,
        ).values.to_cpu().to_vec();

        assert_eq!(gpu_result, cpu_result);
    }
}