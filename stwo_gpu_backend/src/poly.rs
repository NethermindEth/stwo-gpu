use stwo_prover::core::{
    backend::Col,
    circle::{CirclePoint, Coset},
    fields::{m31::BaseField, qm31::SecureField},
    poly::{
        circle::{CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps},
        twiddles::TwiddleTree,
        BitReversedOrder,
    },
};

use crate::{backend::CudaBackend, cuda::BaseFieldCudaColumn};

impl PolyOps for CudaBackend {
    type Twiddles = BaseFieldCudaColumn;

    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        todo!()
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
