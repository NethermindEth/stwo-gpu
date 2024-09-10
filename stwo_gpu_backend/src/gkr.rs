use stwo_prover::core::{
    fields::{m31::BaseField, qm31::SecureField},
    lookups::{
        gkr_prover::GkrOps,
        mle::{Mle, MleOps},
    },
};

use crate::CudaBackend;

impl MleOps<BaseField> for CudaBackend {
    fn fix_first_variable(
        _mle: Mle<Self, BaseField>,
        _assignment: SecureField,
    ) -> Mle<Self, SecureField>
    where
        Self: MleOps<SecureField>,
    {
        todo!()
    }
}

impl MleOps<SecureField> for CudaBackend {
    fn fix_first_variable(
        _mle: Mle<Self, SecureField>,
        _assignment: SecureField,
    ) -> Mle<Self, SecureField>
    where
        Self: MleOps<SecureField>,
    {
        todo!()
    }
}

impl GkrOps for CudaBackend {
    fn gen_eq_evals(_y: &[SecureField], _v: SecureField) -> Mle<Self, SecureField> {
        todo!()
    }

    fn next_layer(
        _layer: &stwo_prover::core::lookups::gkr_prover::Layer<Self>,
    ) -> stwo_prover::core::lookups::gkr_prover::Layer<Self> {
        todo!()
    }

    fn sum_as_poly_in_first_variable(
        _h: &stwo_prover::core::lookups::gkr_prover::GkrMultivariatePolyOracle<'_, Self>,
        _claim: SecureField,
    ) -> stwo_prover::core::lookups::utils::UnivariatePoly<SecureField> {
        todo!()
    }
}
