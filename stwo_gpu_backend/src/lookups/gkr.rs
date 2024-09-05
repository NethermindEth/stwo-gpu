use stwo_prover::core::{
    fields::qm31::SecureField,
    lookups::{
        gkr_prover::GkrOps, mle::Mle,
    },
};

use crate::CudaBackend;

impl GkrOps for CudaBackend {
    fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Mle<Self, SecureField> {
        todo!()
    }

    fn next_layer(
        layer: &stwo_prover::core::lookups::gkr_prover::Layer<Self>,
    ) -> stwo_prover::core::lookups::gkr_prover::Layer<Self> {
        todo!()
    }

    fn sum_as_poly_in_first_variable(
        h: &stwo_prover::core::lookups::gkr_prover::GkrMultivariatePolyOracle<'_, Self>,
        claim: SecureField,
    ) -> stwo_prover::core::lookups::utils::UnivariatePoly<SecureField> {
        todo!()
    }
}
