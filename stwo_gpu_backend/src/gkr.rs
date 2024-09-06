use crate::CudaBackend;
use stwo_prover::core::{
    fields::{m31::BaseField, qm31::SecureField},
    lookups::{
        gkr_prover::GkrOps,
        mle::{Mle, MleOps},
    },
};
use crate::cuda::SecureFieldVec;

impl MleOps<BaseField> for CudaBackend {
    fn fix_first_variable(
        mle: Mle<Self, BaseField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField>
    where
        Self: MleOps<SecureField>,
    {
        todo!()
    }
}

impl MleOps<SecureField> for CudaBackend {
    fn fix_first_variable(
        mle: Mle<Self, SecureField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField>
    where
        Self: MleOps<SecureField>,
    {
        todo!()
    }
}

impl GkrOps for CudaBackend {
    fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Mle<Self, SecureField> {
        let mut evals = Vec::with_capacity(1 << y.len());
        evals.push(v);

        for &y_i in y.iter().rev() {
            for j in 0..evals.len() {
                // `lhs[j] = eq(0, y_i) * c[i]`
                // `rhs[j] = eq(1, y_i) * c[i]`
                let tmp = evals[j] * y_i;
                evals.push(tmp);
                evals[j] -= tmp;
            }
        }
        Mle::new(SecureFieldVec::from_vec(evals))
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

mod tests {
    use crate::CudaBackend;
    use stwo_prover::core::backend::{Column, CpuBackend};
    use stwo_prover::core::fields::m31::BaseField;
    use stwo_prover::core::lookups::gkr_prover::GkrOps;

    #[test]
    fn gen_eq_evals_matches_cpu() {
        let two = BaseField::from(2).into();
        let y = [7, 3, 5, 6, 1, 1, 9].map(|v| BaseField::from(v).into());
        let eq_evals_cpu = CpuBackend::gen_eq_evals(&y, two);

        let eq_evals_gpu = CudaBackend::gen_eq_evals(&y, two);

        assert_eq!(eq_evals_gpu.to_cpu(), *eq_evals_cpu);
    }
}