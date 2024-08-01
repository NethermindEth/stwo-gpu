use itertools::Itertools;
use stwo_prover::core::{
    backend::{Column, CpuBackend},
    fields::{m31::BaseField, qm31::SecureField, secure_column::SecureColumnByCoords},
    pcs::quotients::{ColumnSampleBatch, QuotientOps},
    poly::{
        circle::{CircleDomain, CircleEvaluation, SecureEvaluation}, BitReversedOrder
    },
};

use crate::{backend::CudaBackend, cuda::{bindings, BaseFieldVec, ColumnSampleBatchVec, VecBaseFieldVec}};

impl QuotientOps for CudaBackend {
    fn accumulate_quotients(
        domain: CircleDomain,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        sample_batches: &[ColumnSampleBatch],
    ) -> SecureEvaluation<Self> {

        let vec_base_field_vec = columns.iter().map(|col| &col.values).collect_vec();
        let d_columns = VecBaseFieldVec::from_vec(vec_base_field_vec);
        
        let d_domain_coset = domain.half_coset;
        //let d_sample_batches = ColumnSampleBatchVec::from(sample_batches); 
        
        let values = unsafe {
            bindings::accumulate_quotients(
                d_domain_coset.initial_index.0, 
                d_domain_coset.step_size.0, 
                d_domain_coset.log_size, 
                d_columns.device_ptr, 
                d_columns.col_size, 
                d_columns.row_size, 
                random_coeff,
                // d_sample_batches.device_ptr,
                // d_sample_batches.size,
            )
        };
        
        // cpu mock
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

    use stwo_prover::core::backend::simd::column::BaseColumn;
    use stwo_prover::core::backend::simd::SimdBackend;
    use stwo_prover::core::backend::{Column, CpuBackend};
    use stwo_prover::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use stwo_prover::core::fields::m31::{BaseField, M31};
    use stwo_prover::core::pcs::quotients::{ColumnSampleBatch, QuotientOps};
    use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation};
    use stwo_prover::core::poly::BitReversedOrder;
    use stwo_prover::core::fields::qm31::QM31;

    use crate::cuda::{self, BaseFieldVec};
    use crate::CudaBackend; 
    #[test]
    fn test_accumulate_quotients() {
        const LOG_SIZE: u32 = 4;
        let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
        let e0: BaseColumn = (0..domain.size()).map(BaseField::from).collect();
        let e1: BaseColumn = (0..domain.size()).map(|i| BaseField::from(2 * i)).collect();
        let columns = vec![
            CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(domain, e0.clone()),
            CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(domain, e1.clone()),
        ];

        let e0: BaseFieldVec = cuda::BaseFieldVec::from_vec(e0.to_cpu());
        let e1: BaseFieldVec = cuda::BaseFieldVec::from_vec(e1.to_cpu());
        let columns_gpu = vec![
            CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(domain, e0),
            CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(domain, e1),
        ];
        let random_coeff = QM31::from_u32_unchecked(1, 2, 3, 4);
        let a = QM31::from_u32_unchecked(3, 6, 9, 12);
        let b = QM31::from_u32_unchecked(4, 8, 12, 16);
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
        )
        .values
        .to_vec();

        let res = CudaBackend::accumulate_quotients(
            domain,
            &columns_gpu.iter().collect_vec(),
            random_coeff,
            &samples,
        )
        .values
        .to_cpu().to_vec();

        //println!("{:?}", cpu_result); 

        assert_eq!(res, cpu_result); 
    }

}