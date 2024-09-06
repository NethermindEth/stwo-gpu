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
    use itertools::Itertools;
    use crate::CudaBackend;
    use stwo_prover::core::backend::{Column, CpuBackend};
    use stwo_prover::core::fields::m31::{BaseField, M31};
    use stwo_prover::core::fields::qm31::SecureField;
    use stwo_prover::core::lookups::gkr_prover::GkrOps;

    #[test]
    fn gen_eq_evals_matches_cpu() {
        let two = BaseField::from(2).into();

        let from_raw = [7, 3, 5, 6, 1, 1, 9].repeat(4);
        let y = from_raw.chunks(4).map(|a|
            SecureField::from_u32_unchecked(a[0], a[1], a[2], a[3])
        ).collect_vec();

        let cpu_eq_evals = CpuBackend::gen_eq_evals(&y, two);
        let gpu_eq_evals = CudaBackend::gen_eq_evals(&y, two);

        assert_eq!(gpu_eq_evals.to_cpu(), *cpu_eq_evals);
    }
}