use stwo_prover::core::{
    fields::qm31::SecureField,
    fri::FriOps,
    poly::{circle::SecureEvaluation, line::LineEvaluation, twiddles::TwiddleTree},
};
use stwo_prover::core::fields::m31::{BaseField, M31};
use stwo_prover::core::fields::secure_column::SecureColumn;
use stwo_prover::core::fri::CIRCLE_TO_LINE_FOLD_STEP;
use crate::backend::CudaBackend;
use crate::cuda::{BaseFieldVec, bindings};

impl FriOps for CudaBackend {
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        unsafe {
            let n = eval.len();
            assert!(n >= 2, "Evaluation too small");

            let remaining_folds = n.ilog2();
            let twiddles_size = twiddles.itwiddles.size;
            let twiddle_offset: usize = twiddles_size - (1 << remaining_folds);

            let folded_values = alloc_secure_column_on_gpu_as_array(n >> 1);

            let eval_values = &eval.values;
            let folded_values1 = [
                &folded_values[0],
                &folded_values[1],
                &folded_values[2],
                &folded_values[3]
            ];
            let gpu_domain = twiddles.itwiddles.device_ptr;

            bindings::fold_circle(gpu_domain, twiddle_offset, n,
                                  eval_values.columns[0].device_ptr,
                                  eval_values.columns[1].device_ptr,
                                  eval_values.columns[2].device_ptr,
                                  eval_values.columns[3].device_ptr,
                                  alpha,
                                  folded_values1[0].device_ptr,
                                  folded_values1[1].device_ptr,
                                  folded_values1[2].device_ptr,
                                  folded_values1[3].device_ptr,
            );

            let folded_values = SecureColumn { columns: folded_values };
            LineEvaluation::new(eval.domain().double(), folded_values)
        }
    }

    fn fold_circle_into_line(
        dst: &mut LineEvaluation<Self>,
        src: &SecureEvaluation<Self>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) {
        unsafe {
            let n = src.len();
            assert_eq!(n >> CIRCLE_TO_LINE_FOLD_STEP, dst.len());

            let folded_values = &dst.values.columns;
            let eval_values = &src.values;
            let folded_values1 = [
                &folded_values[0],
                &folded_values[1],
                &folded_values[2],
                &folded_values[3]
            ];
            let gpu_domain = twiddles.itwiddles.device_ptr;

            bindings::fold_circle_into_line(gpu_domain, 0, n,
                                            eval_values.columns[0].device_ptr,
                                            eval_values.columns[1].device_ptr,
                                            eval_values.columns[2].device_ptr,
                                            eval_values.columns[3].device_ptr,
                                            alpha,
                                            folded_values1[0].device_ptr,
                                            folded_values1[1].device_ptr,
                                            folded_values1[2].device_ptr,
                                            folded_values1[3].device_ptr,
            );
        }
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

unsafe fn launch_kernel_for_fold(
    eval_values: &SecureColumn<CudaBackend>,
    twiddles: &TwiddleTree<CudaBackend>,
    twiddle_offset: usize,
    folded_values: [&BaseFieldVec; 4],
    alpha: SecureField,
    n: usize) {
    let gpu_domain = twiddles.itwiddles.device_ptr;

    bindings::fold_circle(gpu_domain, twiddle_offset, n,
                          eval_values.columns[0].device_ptr,
                          eval_values.columns[1].device_ptr,
                          eval_values.columns[2].device_ptr,
                          eval_values.columns[3].device_ptr,
                          alpha,
                          folded_values[0].device_ptr,
                          folded_values[1].device_ptr,
                          folded_values[2].device_ptr,
                          folded_values[3].device_ptr,
    );
}

unsafe fn alloc_secure_column_on_gpu_as_array(n: usize) -> [BaseFieldVec; 4] {
    let folded_values_0 = BaseFieldVec::new_zeroes(n);
    let folded_values_1 = BaseFieldVec::new_zeroes(n);
    let folded_values_2 = BaseFieldVec::new_zeroes(n);
    let folded_values_3 = BaseFieldVec::new_zeroes(n);

    [folded_values_0, folded_values_1, folded_values_2, folded_values_3]
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
    use std::iter::zip;
    use itertools::Itertools;
    use rand::{Rng, SeedableRng};
    use rand::rngs::SmallRng;
    use stwo_prover::core::backend::{Column, ColumnOps, CpuBackend};
    use stwo_prover::core::circle::Coset;
    use stwo_prover::core::fields::Field;
    use stwo_prover::core::fields::m31::{BaseField, M31};
    use stwo_prover::core::fields::qm31::SecureField;
    use stwo_prover::core::fields::secure_column::SecureColumn;
    use stwo_prover::core::fri::FriOps;
    use stwo_prover::core::poly::circle::{CanonicCoset, PolyOps, SecureEvaluation};
    use stwo_prover::core::poly::line::{LineDomain, LineEvaluation, LinePoly};

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
                vec[0].push(M31::from_u32_unchecked(a[0]));
                vec[1].push(M31::from_u32_unchecked(a[1]));
                vec[2].push(M31::from_u32_unchecked(a[2]));
                vec[3].push(M31::from_u32_unchecked(a[3]));
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

    #[test]
    fn test_fold_line_compared_with_cpu() {
        const LOG_SIZE: u32 = 20;
        let mut rng = SmallRng::seed_from_u64(0);
        let values: Vec<SecureField> = (0..1 << LOG_SIZE).map(|_| rng.gen()).collect_vec();
        let alpha = SecureField::from_u32_unchecked(1, 3, 5, 7);
        let domain = LineDomain::new(CanonicCoset::new(LOG_SIZE + 1).half_coset());

        let mut vec: [Vec<BaseField>; 4] = [vec!(), vec!(), vec!(), vec!()];
        values.iter()
            .for_each(|a| {
                vec[0].push(BaseField::from_u32_unchecked(a.0.0.0));
                vec[1].push(BaseField::from_u32_unchecked(a.0.1.0));
                vec[2].push(BaseField::from_u32_unchecked(a.1.0.0));
                vec[3].push(BaseField::from_u32_unchecked(a.1.1.0));
            });

        let cpu_fold = CpuBackend::fold_line(
            &LineEvaluation::new(domain, SecureColumn { columns: vec.clone() }),
            alpha,
            &CpuBackend::precompute_twiddles(domain.coset()),
        );
        let vecs = [
            BaseFieldVec::from_vec(vec[0].clone()),
            BaseFieldVec::from_vec(vec[1].clone()),
            BaseFieldVec::from_vec(vec[2].clone()),
            BaseFieldVec::from_vec(vec[3].clone())];

        let gpu_fold = CudaBackend::fold_line(
            &LineEvaluation::new(domain, SecureColumn { columns: vecs }),
            alpha,
            &CudaBackend::precompute_twiddles(domain.coset()),
        );

        assert_eq!(cpu_fold.values.to_vec(), gpu_fold.values.to_cpu().to_vec());
    }

    #[test]
    fn test_fold_line() {
        const DEGREE: usize = 8;
        let even_coeffs: [SecureField; DEGREE / 2] = [1, 2, 1, 3]
            .map(BaseField::from_u32_unchecked)
            .map(SecureField::from);
        let odd_coeffs: [SecureField; DEGREE / 2] = [3, 5, 4, 1]
            .map(BaseField::from_u32_unchecked)
            .map(SecureField::from);
        let poly = LinePoly::new([even_coeffs, odd_coeffs].concat());
        let even_poly = LinePoly::new(even_coeffs.to_vec());
        let odd_poly = LinePoly::new(odd_coeffs.to_vec());
        let alpha = BaseField::from_u32_unchecked(19283).into();
        let domain = LineDomain::new(Coset::half_odds(DEGREE.ilog2()));
        let drp_domain = domain.double();
        let mut values: Vec<SecureField> = domain
            .iter()
            .map(|p| poly.eval_at_point(p.into()))
            .collect();
        CpuBackend::bit_reverse_column(&mut values);
        let mut vec: [Vec<BaseField>; 4] = [vec!(), vec!(), vec!(), vec!()];
        values.iter()
            .for_each(|a| {
                vec[0].push(BaseField::from_u32_unchecked(a.0.0.0));
                vec[1].push(BaseField::from_u32_unchecked(a.0.1.0));
                vec[2].push(BaseField::from_u32_unchecked(a.1.0.0));
                vec[3].push(BaseField::from_u32_unchecked(a.1.1.0));
            });
        let vecs = [
            BaseFieldVec::from_vec(vec[0].clone()),
            BaseFieldVec::from_vec(vec[1].clone()),
            BaseFieldVec::from_vec(vec[2].clone()),
            BaseFieldVec::from_vec(vec[3].clone())];
        let evals: LineEvaluation<CudaBackend> = LineEvaluation::new(domain, SecureColumn { columns: vecs });

        let drp_evals = CudaBackend::fold_line(&evals, alpha, &CudaBackend::precompute_twiddles(domain.coset()));
        let mut drp_evals = drp_evals.values.to_cpu().into_iter().collect_vec();
        CpuBackend::bit_reverse_column(&mut drp_evals);

        assert_eq!(drp_evals.len(), DEGREE / 2);
        for (i, (&drp_eval, x)) in zip(&drp_evals, drp_domain).enumerate() {
            let f_e: SecureField = even_poly.eval_at_point(x.into());
            let f_o: SecureField = odd_poly.eval_at_point(x.into());

            assert_eq!(drp_eval, (f_e + alpha * f_o).double(), "mismatch at {i}");
        }
    }

    #[test]
    fn test_fold_line_more_than_once() {
        const LOG_SIZE: u32 = 13;
        let mut rng = SmallRng::seed_from_u64(0);
        let values: Vec<SecureField> = (0..1 << LOG_SIZE).map(|_| rng.gen()).collect_vec();
        let alpha = SecureField::from_u32_unchecked(1, 3, 5, 7);
        let domain = LineDomain::new(CanonicCoset::new(LOG_SIZE + 1).half_coset());

        let mut vec: [Vec<BaseField>; 4] = [vec!(), vec!(), vec!(), vec!()];
        values.iter()
            .for_each(|a| {
                vec[0].push(BaseField::from_u32_unchecked(a.0.0.0));
                vec[1].push(BaseField::from_u32_unchecked(a.0.1.0));
                vec[2].push(BaseField::from_u32_unchecked(a.1.0.0));
                vec[3].push(BaseField::from_u32_unchecked(a.1.1.0));
            });
        let vecs = [
            BaseFieldVec::from_vec(vec[0].clone()),
            BaseFieldVec::from_vec(vec[1].clone()),
            BaseFieldVec::from_vec(vec[2].clone()),
            BaseFieldVec::from_vec(vec[3].clone())];

        let first_cpu_fold = CpuBackend::fold_line(
            &LineEvaluation::new(domain, SecureColumn { columns: vec.clone() }),
            alpha,
            &CpuBackend::precompute_twiddles(domain.coset()),
        );
        let second_cpu_fold = CpuBackend::fold_line(
            &first_cpu_fold,
            alpha,
            &CpuBackend::precompute_twiddles(domain.coset()),
        );
        let third_cpu_fold = CpuBackend::fold_line(
            &second_cpu_fold,
            alpha,
            &CpuBackend::precompute_twiddles(domain.coset()),
        );

        let first_gpu_fold = CudaBackend::fold_line(
            &LineEvaluation::new(domain, SecureColumn { columns: vecs }),
            alpha,
            &CudaBackend::precompute_twiddles(domain.coset()),
        );
        let second_gpu_fold = CudaBackend::fold_line(
            &first_gpu_fold,
            alpha,
            &CudaBackend::precompute_twiddles(domain.coset()),
        );
        let third_gpu_fold = CudaBackend::fold_line(
            &second_gpu_fold,
            alpha,
            &CudaBackend::precompute_twiddles(domain.coset()),
        );

        assert_eq!(third_cpu_fold.values.to_vec(), third_gpu_fold.values.to_cpu().to_vec());
    }


    #[test]
    fn test_fold_circle_into_line_compared_with_cpu() {
        const LOG_SIZE: u32 = 13;
        let values: Vec<SecureField> = (0..(1 << LOG_SIZE))
            .map(|i| SecureField::from_u32_unchecked(4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3))
            .collect();
        let alpha = SecureField::from_u32_unchecked(1, 3, 5, 7);
        let circle_domain = CanonicCoset::new(LOG_SIZE).circle_domain();
        let line_domain = LineDomain::new(circle_domain.half_coset);
        let mut cpu_fold =
            LineEvaluation::new(line_domain, SecureColumn::zeros(1 << (LOG_SIZE - 1)));

        let mut vec: [Vec<BaseField>; 4] = [vec!(), vec!(), vec!(), vec!()];
        values.iter()
            .for_each(|a| {
                vec[0].push(BaseField::from_u32_unchecked(a.0.0.0));
                vec[1].push(BaseField::from_u32_unchecked(a.0.1.0));
                vec[2].push(BaseField::from_u32_unchecked(a.1.0.0));
                vec[3].push(BaseField::from_u32_unchecked(a.1.1.0));
            });
        let vecs = [
            BaseFieldVec::from_vec(vec[0].clone()),
            BaseFieldVec::from_vec(vec[1].clone()),
            BaseFieldVec::from_vec(vec[2].clone()),
            BaseFieldVec::from_vec(vec[3].clone())];
        CpuBackend::fold_circle_into_line(
            &mut cpu_fold,
            &SecureEvaluation {
                domain: circle_domain,
                values: SecureColumn { columns: vec.clone() },
            },
            alpha,
            &CpuBackend::precompute_twiddles(line_domain.coset()),
        );

        let mut cuda_fold =
            LineEvaluation::new(line_domain, SecureColumn::zeros(1 << (LOG_SIZE - 1)));
        CudaBackend::fold_circle_into_line(
            &mut cuda_fold,
            &SecureEvaluation {
                domain: circle_domain,
                values: SecureColumn { columns: vecs },
            },
            alpha,
            &CudaBackend::precompute_twiddles(line_domain.coset()),
        );

        assert_eq!(cpu_fold.values.to_vec(), cuda_fold.values.to_cpu().to_vec());
    }
}