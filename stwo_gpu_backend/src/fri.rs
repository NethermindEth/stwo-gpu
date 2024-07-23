use stwo_prover::core::{
    fields::qm31::SecureField,
    fri::FriOps,
    poly::{circle::SecureEvaluation, line::LineEvaluation, twiddles::TwiddleTree},
};

use crate::backend::CudaBackend;

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

    fn decompose(_eval: &SecureEvaluation<Self>) -> (SecureEvaluation<Self>, SecureField) {
        todo!()
    }
}
