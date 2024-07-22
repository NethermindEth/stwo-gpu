use stwo_prover::core::{
    fields::qm31::SecureField,
    fri::FriOps,
    poly::{circle::SecureEvaluation, line::LineEvaluation, twiddles::TwiddleTree},
};
use stwo_prover::core::fields::m31::M31;
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
                columns: eval.columns.clone()
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
    unsafe fn sum(column: &BaseFieldVec) -> M31 {
        let column_size = column.size;
        let mut temp = BaseFieldVec::new_uninitialized(column_size);

        let mut partial_results = BaseFieldVec::new_uninitialized(1);
        unsafe {
            bindings::sum(column.device_ptr,
                          temp.device_ptr,
                          partial_results.device_ptr,
                          column_size as u32);
        }

        let result = partial_results.to_vec()[0];

        return result;
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use stwo_prover::core::backend::CpuBackend;
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

        let (_, expected_lambda) = CpuBackend::decompose(&cpu_secure_evaluation);
        let (_, lambda) = CudaBackend::decompose(&gpu_secure_evaluation);

        assert_eq!(lambda, expected_lambda);
    }

    #[test]
    fn test_decompose_using_less_than_an_entire_block() {
        test_decompose_with_domain_log_size(5);
    }
}