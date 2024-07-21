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
    cuda::{self, bindings::batch_inverse_base_field},
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
        let result = cuda::BaseFieldVec { device_ptr, size };
        CircleEvaluation::new(coset.circle_domain(), result)
    }

    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        itwiddles: &TwiddleTree<Self>,
    ) -> CirclePoly<Self> {
        todo!()
    }

    fn eval_at_point(poly: &CirclePoly<Self>, point: CirclePoint<SecureField>) -> SecureField {
        todo!()
    }

    fn extend(poly: &CirclePoly<Self>, log_size: u32) -> CirclePoly<Self> {
        todo!()
    }

    fn evaluate(
        poly: &CirclePoly<Self>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        todo!()
    }

    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self> {
        unsafe {
            let twiddles_device_ptr = cuda::bindings::precompute_twiddles(
                coset.initial.into(),
                coset.step.into(),
                coset.size(),
            );
            let twiddles = cuda::BaseFieldVec {
                device_ptr: twiddles_device_ptr,
                size: coset.size(),
            };
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
        fields::m31::BaseField,
        poly::circle::{CanonicCoset, PolyOps},
    };

    #[test]
    fn test_new_canonical_ordered() {
        let log_size = 25;
        let coset = CanonicCoset::new(log_size);
        let size: usize = 1 << log_size;
        let column_data = (0..size as u32)
            .map(|x| BaseField::from(x))
            .collect::<Vec<_>>();
        let cpu_values = column_data.clone();
        let expected_result = CpuBackend::new_canonical_ordered(coset.clone(), cpu_values);

        let column = cuda::BaseFieldVec::new(column_data);
        let result = CudaBackend::new_canonical_ordered(coset, column);

        assert_eq!(result.values.to_cpu(), expected_result.values);
        assert_eq!(
            result.domain.iter().collect::<Vec<_>>(),
            expected_result.domain.iter().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_precompute_twiddles() {
        let log_size = 20;

        let half_coset = CanonicCoset::new(log_size).half_coset();
        let expected_result = CpuBackend::precompute_twiddles(half_coset.clone());
        let twiddles = CudaBackend::precompute_twiddles(half_coset);

        assert_eq!(twiddles.twiddles.to_cpu(), expected_result.twiddles);
        assert_eq!(twiddles.itwiddles.to_cpu(), expected_result.itwiddles);
        assert_eq!(
            twiddles.root_coset.iter().collect::<Vec<_>>(),
            expected_result.root_coset.iter().collect::<Vec<_>>()
        );
    }
}
