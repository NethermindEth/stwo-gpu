use stwo_prover::core::{
    backend::{simd::bit_reverse, Col, Column, ColumnOps},
    circle::{CirclePoint, Coset},
    fields::{m31::BaseField, qm31::SecureField},
    poly::{
        circle::{CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps},
        twiddles::TwiddleTree,
        BitReversedOrder,
    },
};

use crate::{backend::CudaBackend, cuda};

impl PolyOps for CudaBackend {
    type Twiddles = cuda::BaseFieldVec;

    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        let mut sorted_values = cuda::BaseFieldVec {
            device_ptr: unsafe { cuda::bindings::sort_values(values.device_ptr, values.len()) },
            size: values.len(),
        };
        <Self as ColumnOps<BaseField>>::bit_reverse_column(&mut sorted_values);
        CircleEvaluation::new(coset.circle_domain(), sorted_values)
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
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use stwo_prover::core::{
        backend::{Column, CpuBackend},
        fields::m31::BaseField,
        poly::circle::{CanonicCoset, PolyOps},
    };

    use crate::{backend::CudaBackend, cuda};

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
}
