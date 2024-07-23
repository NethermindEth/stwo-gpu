use stwo_prover::core::{
    fields::qm31::SecureField,
    fri::FriOps,
    poly::{circle::SecureEvaluation, line::LineEvaluation, twiddles::TwiddleTree},
};
use stwo_prover::core::fields::m31::{BaseField, M31};
use stwo_prover::core::fields::secure_column::SecureColumn;

use crate::backend::CudaBackend;
use crate::cuda::{BaseFieldVec, bindings};

impl FriOps for CudaBackend {
    fn fold_line(
        _eval: &LineEvaluation<Self>,
        _alpha: SecureField,
        _twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        todo!()
    }

    fn fold_circle_into_line(
        _dst: &mut LineEvaluation<Self>,
        _src: &SecureEvaluation<Self>,
        _alpha: SecureField,
        _twiddles: &TwiddleTree<Self>,
    ) {
        todo!()
    }

    fn decompose(eval: &SecureEvaluation<Self>) -> (SecureEvaluation<Self>, SecureField) {
        let columns = &eval.columns;

        let lambda = unsafe {
            let a: M31 = Self::sum(&columns[0]);
            let b = Self::sum(&columns[1]);
            let c = Self::sum(&columns[2]);
            let d = Self::sum(&columns[3]);
            SecureField::from_m31(a, b, c, d) / M31::from_u32_unchecked(eval.len() as u32)
        };

        let g_values = unsafe {
            SecureColumn {
                columns: [
                    Self::compute_g_values(&columns[0], lambda.0.0),
                    Self::compute_g_values(&columns[1], lambda.0.1),
                    Self::compute_g_values(&columns[2], lambda.1.0),
                    Self::compute_g_values(&columns[3], lambda.1.1),
                ]
            }
        };

        let g = SecureEvaluation {
            domain: eval.domain,
            values: g_values,
        };
        (g, lambda)
    }
}

impl CudaBackend {
    unsafe fn sum(column: &BaseFieldVec) -> BaseField {
        let column_size = column.size;
        return bindings::sum(column.device_ptr,
                             column_size as u32);
    }

    unsafe fn compute_g_values(f_values: &BaseFieldVec, lambda: M31) -> BaseFieldVec {
        let size = f_values.size;

        let result = BaseFieldVec {
            device_ptr: bindings::compute_g_values(
                f_values.device_ptr,
                size,
                lambda),
            size: size,
        };
        return result;
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use stwo_prover::core::backend::{Column, CpuBackend};
    use stwo_prover::core::fields::m31::M31;
    use stwo_prover::core::fields::secure_column::SecureColumn;
    use stwo_prover::core::fri::FriOps;
    use stwo_prover::core::poly::circle::{CanonicCoset, SecureEvaluation};

    use crate::backend::CudaBackend;
    use crate::cuda::BaseFieldVec;

    fn test_decompose_with_domain_log_size(domain_log_size: u32) {
        let size = 1 << domain_log_size;
        let coset = CanonicCoset::new(domain_log_size);
        let domain = coset.circle_domain();

        let from_raw = (0..size * 4 as u32).collect::<Vec<u32>>();
        let mut vec: [Vec<M31>; 4] = [vec!(), vec!(), vec!(), vec!()];

        from_raw
            .chunks(4)
            .for_each(|a| {
                &vec[0].push(M31::from_u32_unchecked(a[0]));
                &vec[1].push(M31::from_u32_unchecked(a[1]));
                &vec[2].push(M31::from_u32_unchecked(a[2]));
                &vec[3].push(M31::from_u32_unchecked(a[3]));
            });

        let cpu_secure_evaluation = SecureEvaluation {
            domain: domain,
            values: SecureColumn { columns: vec.clone() },
        };

        let columns = [
            BaseFieldVec::from_vec(vec[0].clone()),
            BaseFieldVec::from_vec(vec[1].clone()),
            BaseFieldVec::from_vec(vec[2].clone()),
            BaseFieldVec::from_vec(vec[3].clone())];
        let gpu_secure_evaluation = SecureEvaluation {
            domain: domain,
            values: SecureColumn { columns },
        };

        let (expected_g_values, expected_lambda) = CpuBackend::decompose(&cpu_secure_evaluation);
        let (g_values, lambda) = CudaBackend::decompose(&gpu_secure_evaluation);

        assert_eq!(lambda, expected_lambda);
        assert_eq!(g_values.values.columns[0].to_cpu(), expected_g_values.values.columns[0]);
        assert_eq!(g_values.values.columns[1].to_cpu(), expected_g_values.values.columns[1]);
        assert_eq!(g_values.values.columns[2].to_cpu(), expected_g_values.values.columns[2]);
        assert_eq!(g_values.values.columns[3].to_cpu(), expected_g_values.values.columns[3]);
    }

    #[test]
    fn test_decompose_using_less_than_an_entire_block() {
        test_decompose_with_domain_log_size(5);
    }

    #[test]
    fn test_decompose_using_an_entire_block() {
        test_decompose_with_domain_log_size(11);
    }

    #[test]
    fn test_decompose_using_more_than_entire_block() {
        test_decompose_with_domain_log_size(11 + 4);
    }

    #[test]
    fn test_decompose_using_an_entire_block_for_results() {
        test_decompose_with_domain_log_size(22);
    }

    #[ignore]
    #[test]
    fn test_decompose_using_more_than_an_entire_block_for_results() {
        test_decompose_with_domain_log_size(27);
    }
}