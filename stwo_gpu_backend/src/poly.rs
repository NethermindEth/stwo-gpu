use stwo_prover::core::{
    backend::{Col, Column},
    circle::{CirclePoint, Coset},
    fields::{m31::BaseField, qm31::SecureField},
    poly::{
        circle::{CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps},
        twiddles::TwiddleTree,
        BitReversedOrder,
    },
};

use crate::{
    backend::CudaBackend,
    cuda::{self},
};

impl PolyOps for CudaBackend {
    type Twiddles = cuda::BaseFieldVec;

    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        let size = values.len();
        let device_ptr = unsafe {
            cuda::bindings::sort_values_and_permute_with_bit_reverse_order(values.device_ptr, size)
        };
        let result = cuda::BaseFieldVec::new(device_ptr, size);
        CircleEvaluation::new(coset.circle_domain(), result)
    }

    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        twiddle_tree: &TwiddleTree<Self>,
    ) -> CirclePoly<Self> {
        let values = eval.values;
        assert!(eval
            .domain
            .half_coset
            .is_doubling_of(twiddle_tree.root_coset));
        unsafe {
            cuda::bindings::interpolate(
                values.device_ptr,
                twiddle_tree.itwiddles.device_ptr,
                values.len() as u32,
            );
        }
        CirclePoly::new(values)
    }

    fn eval_at_point(poly: &CirclePoly<Self>, point: CirclePoint<SecureField>) -> SecureField {
        unsafe {
            cuda::bindings::eval_at_point(
                poly.coeffs.device_ptr,
                poly.coeffs.len() as u32,
                point.x,
                point.y,
            )
        }
    }

    fn extend(poly: &CirclePoly<Self>, log_size: u32) -> CirclePoly<Self> {
        let new_size = 1 << log_size;
        assert!(
            new_size >= poly.coeffs.len(),
            "New size must be larger than the old size"
        );

        let mut new_coeffs = cuda::BaseFieldVec::new_zeroes(new_size);
        new_coeffs.copy_from(&poly.coeffs);
        CirclePoly::new(new_coeffs)
    }

    fn evaluate(
        poly: &CirclePoly<Self>,
        domain: CircleDomain,
        twiddle_tree: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        let values = poly.extend(domain.log_size()).coeffs;
        assert!(domain.half_coset.is_doubling_of(twiddle_tree.root_coset));
        unsafe {
            cuda::bindings::evaluate(
                values.device_ptr,
                twiddle_tree.twiddles.device_ptr,
                values.len() as u32,
            );
        }

        CircleEvaluation::new(domain, values)
    }

    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self> {
        unsafe {
            let twiddles = cuda::BaseFieldVec::new(
                cuda::bindings::precompute_twiddles(
                    coset.initial.into(),
                    coset.step.into(),
                    coset.size(),
                ),
                coset.size(),
            );
            let itwiddles = cuda::BaseFieldVec::new_uninitialized(coset.size());
            cuda::bindings::batch_inverse_base_field(
                twiddles.device_ptr,
                itwiddles.device_ptr,
                coset.size(),
            );
            TwiddleTree {
                root_coset: coset,
                twiddles,
                itwiddles,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{backend::CudaBackend, cuda};
    use stwo_prover::core::{
        backend::{Column, CpuBackend},
        circle::SECURE_FIELD_CIRCLE_GEN,
        fields::m31::BaseField,
        poly::circle::{CanonicCoset, CirclePoly, PolyOps},
    };

    #[test]
    fn test_new_canonical_ordered() {
        let log_size = 23;
        let coset = CanonicCoset::new(log_size);
        let size: usize = 1 << log_size;
        let column_data = (0..size as u32).map(BaseField::from).collect::<Vec<_>>();
        let cpu_values = column_data.clone();
        let expected_result = CpuBackend::new_canonical_ordered(coset, cpu_values);

        let column = cuda::BaseFieldVec::from_vec(column_data);
        let result = CudaBackend::new_canonical_ordered(coset, column);

        assert_eq!(result.values.to_cpu(), expected_result.values);
        assert_eq!(
            result.domain.iter().collect::<Vec<_>>(),
            expected_result.domain.iter().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_precompute_twiddles() {
        let log_size = 3;

        let half_coset = CanonicCoset::new(log_size).half_coset();
        let expected_result = CpuBackend::precompute_twiddles(half_coset);
        let twiddles = CudaBackend::precompute_twiddles(half_coset);

        assert_eq!(twiddles.twiddles.to_cpu(), expected_result.twiddles);
        assert_eq!(twiddles.itwiddles.to_cpu(), expected_result.itwiddles);
        assert_eq!(
            twiddles.root_coset.iter().collect::<Vec<_>>(),
            expected_result.root_coset.iter().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_extend() {
        let log_size = 20;
        let size = 1 << log_size;
        let new_log_size = log_size + 5;
        let cpu_coeffs = (0..size).map(BaseField::from).collect::<Vec<_>>();
        let cuda_coeffs = cuda::BaseFieldVec::from_vec(cpu_coeffs.clone());
        let cpu_poly = CirclePoly::<CpuBackend>::new(cpu_coeffs);
        let cuda_poly = CirclePoly::<CudaBackend>::new(cuda_coeffs);
        let result = CudaBackend::extend(&cuda_poly, new_log_size);
        let expected_result = CpuBackend::extend(&cpu_poly, new_log_size);
        assert_eq!(result.coeffs.to_cpu(), expected_result.coeffs);
        assert_eq!(result.log_size(), expected_result.log_size());
    }

    #[test]
    fn test_interpolate() {
        let log_size = 20;

        let size = 1 << log_size;

        let cpu_values = (1..(size + 1) as u32)
            .map(BaseField::from)
            .collect::<Vec<_>>();
        let gpu_values = cuda::BaseFieldVec::from_vec(cpu_values.clone());

        let coset = CanonicCoset::new(log_size);
        let cpu_evaluations = CpuBackend::new_canonical_ordered(coset, cpu_values);
        let gpu_evaluations = CudaBackend::new_canonical_ordered(coset, gpu_values);

        let cpu_twiddles = CpuBackend::precompute_twiddles(coset.half_coset());
        let gpu_twiddles = CudaBackend::precompute_twiddles(coset.half_coset());

        let expected_result = CpuBackend::interpolate(cpu_evaluations, &cpu_twiddles);
        let result = CudaBackend::interpolate(gpu_evaluations, &gpu_twiddles);

        assert_eq!(result.coeffs.to_cpu(), expected_result.coeffs);
    }

    #[test]
    fn test_evaluate() {
        let log_size = 20;

        let size = 1 << log_size;

        let cpu_values = (1..(size + 1) as u32)
            .map(BaseField::from)
            .collect::<Vec<_>>();
        let gpu_values = cuda::BaseFieldVec::from_vec(cpu_values.clone());

        let coset = CanonicCoset::new(log_size);
        let cpu_evaluations = CpuBackend::new_canonical_ordered(coset, cpu_values);
        let gpu_evaluations = CudaBackend::new_canonical_ordered(coset, gpu_values);

        let cpu_twiddles = CpuBackend::precompute_twiddles(coset.half_coset());
        let gpu_twiddles = CudaBackend::precompute_twiddles(coset.half_coset());

        let cpu_poly = CpuBackend::interpolate(cpu_evaluations, &cpu_twiddles);
        let gpu_poly = CudaBackend::interpolate(gpu_evaluations, &gpu_twiddles);

        let expected_result = CpuBackend::evaluate(&cpu_poly, coset.circle_domain(), &cpu_twiddles);
        let result = CudaBackend::evaluate(&gpu_poly, coset.circle_domain(), &gpu_twiddles);

        assert_eq!(result.values.to_cpu(), expected_result.values);
    }

    #[test]
    fn test_eval_at_point() {
        let log_size = 25;

        let size = 1 << log_size;
        let coset = CanonicCoset::new(log_size);
        let point = SECURE_FIELD_CIRCLE_GEN;

        let cpu_values = (1..(size + 1) as u32)
            .map(BaseField::from)
            .collect::<Vec<_>>();

        let gpu_values = cuda::BaseFieldVec::from_vec(cpu_values.clone());
        let gpu_evaluations = CudaBackend::new_canonical_ordered(coset, gpu_values);
        let gpu_twiddles = CudaBackend::precompute_twiddles(coset.half_coset());
        let gpu_poly = CudaBackend::interpolate(gpu_evaluations, &gpu_twiddles);
        let result = CudaBackend::eval_at_point(&gpu_poly, point);

        let cpu_evaluations = CpuBackend::new_canonical_ordered(coset, cpu_values);
        let cpu_twiddles = CpuBackend::precompute_twiddles(coset.half_coset());
        let cpu_poly = CpuBackend::interpolate(cpu_evaluations, &cpu_twiddles);

        let expected_result = CpuBackend::eval_at_point(&cpu_poly, point.clone());

        assert_eq!(result, expected_result);
    }
}
