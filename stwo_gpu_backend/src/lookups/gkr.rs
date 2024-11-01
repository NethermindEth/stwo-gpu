use crate::CudaBackend;
use stwo_prover::core::{
    fields::{m31::BaseField, qm31::SecureField},
    lookups::{
        gkr_prover::{GkrOps, Layer},
        mle::{Mle, MleOps},
    },
};
use crate::cuda::{bindings, SecureFieldVec};
use crate::cuda::bindings::CudaSecureField;

impl GkrOps for CudaBackend {
    fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Mle<Self, SecureField> {
        let y_size = y.len();
        let mut result_evals = SecureFieldVec::new_uninitialized(1 << y_size);

        unsafe {
            bindings::gen_eq_evals(
                v.into(),
                y.as_ptr() as *const CudaSecureField,
                y_size as u32,
                result_evals.device_ptr as *const CudaSecureField,
                result_evals.size as u32,
            );
        }

        Mle::new(result_evals)
    }

    fn next_layer(
        layer: &stwo_prover::core::lookups::gkr_prover::Layer<Self>,
    ) -> stwo_prover::core::lookups::gkr_prover::Layer<Self> {
        match layer {
            Layer::GrandProduct(col) => next_grand_product_layer(col),
            Layer::LogUpGeneric {
                numerators,
                denominators,
            } => next_logup_generic_layer(numerators, denominators),
            Layer::LogUpMultiplicities {
                numerators,
                denominators,
            } => next_logup_multiplicities_layer(numerators, denominators),
            Layer::LogUpSingles { denominators } => next_logup_singles_layer(denominators),
        }
    }

    fn sum_as_poly_in_first_variable(
        h: &stwo_prover::core::lookups::gkr_prover::GkrMultivariatePolyOracle<'_, Self>,
        claim: SecureField,
    ) -> stwo_prover::core::lookups::utils::UnivariatePoly<SecureField> {
        todo!()
    }
}

fn next_grand_product_layer(layer: &Mle<CudaBackend, SecureField>) -> Layer<CpuBackend> {
    let next_layer_size = layer.size / 2;
    let next_layer = SecureFieldVec::new_uninitialized(next_layer_size); 
    unsafe {
        bindings::next_grand_product_layer(layer.device_ptr, layer.size, next_layer.device_ptr, next_layer_size);
    }
    Layer::GrandProduct(Mle::new(next_layer))
}

fn next_logup_generic_layer<F>(
    numerators: MleExpr<'_, F>,
    denominators: &Mle<CudaBackend, SecureField>,
) -> Layer<CudaBackend>
where
    F: Field,
    SecureField: ExtensionOf<F>,
    CudaBackend: MleOps<F>,
{
    let next_layer_len = denominators.len() / 2; 
    let next_numerators = SecureFieldVec::new_uninitialized(next_layer_len);
    let next_denominators = SecureFieldVec::new_uninitialized(next_layer_len);

    
    for i in 0..half_n {
        let a = Fraction::new(numerators[i * 2], denominators[i * 2]);
        let b = Fraction::new(numerators[i * 2 + 1], denominators[i * 2 + 1]);
        let res = a + b;
        next_numerators.push(res.numerator);
        next_denominators.push(res.denominator);
    }

    Layer::LogUpGeneric {
        numerators: Mle::new(next_numerators),
        denominators: Mle::new(next_denominators),
    }
}

mod tests {
    use itertools::Itertools;
    use crate::CudaBackend;
    use stwo_prover::core::backend::{Column, CpuBackend};
    use stwo_prover::core::fields::m31::{BaseField, M31};
    use stwo_prover::core::fields::qm31::SecureField;
    use stwo_prover::core::lookups::gkr_prover::GkrOps;

    
    use stwo_prover::core::backend::simd::SimdBackend;
    // use stwo_prover::core::backend::{Column, CpuBackend};
    // use stwo_prover::core::channel::Channel;
    // use stwo_prover::core::fields::m31::BaseField;
    // use stwo_prover::core::fields::qm31::SecureField;
    use stwo_prover::core::lookups::gkr_prover::{prove_batch, Layer};
    use stwo_prover::core::lookups::gkr_verifier::{partially_verify_batch, Gate, GkrArtifact, GkrError};
    use stwo_prover::core::lookups::mle::Mle;
    use stwo_prover::core::lookups::utils::Fraction;
    use stwo_prover::core::channel::{Blake2sChannel, Channel};

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

    // #[test]
    // fn grand_product_works() {
    //     const N: usize = 1 << 8;
    //     let values = Blake2sChannel::default().draw_felts(N);
    //     let product = values.iter().product();
    //     let col = Mle::<CudaBackend, SecureField>::new(values.into_iter().collect());
    //     let input_layer = Layer::GrandProduct(col.clone());
    //     let (proof, _) = prove_batch(&mut Blake2sChannel::default(), vec![input_layer]);

    //     let GkrArtifact {
    //         ood_point,
    //         claims_to_verify_by_instance,
    //         n_variables_by_instance: _,
    //     } = partially_verify_batch(vec![Gate::GrandProduct], &proof, &mut Blake2sChannel::default()).unwrap();

    //     assert_eq!(proof.output_claims_by_instance, [vec![product]]);
    //     assert_eq!(
    //         claims_to_verify_by_instance,
    //         [vec![eval_at_point(&col, &ood_point)]]
    //     );
    // }

    pub(crate) fn eval_at_point(input: &Mle<SimdBackend, SecureField>, point: &[SecureField]) -> SecureField {
        pub fn eval(mle_evals: &[SecureField], p: &[SecureField]) -> SecureField {
            match p {
                [] => mle_evals[0],
                &[p_i, ref p @ ..] => {
                    let (lhs, rhs) = mle_evals.split_at(mle_evals.len() / 2);
                    let lhs_eval = eval(lhs, p);
                    let rhs_eval = eval(rhs, p);
                    // Equivalent to `eq(0, p_i) * lhs_eval + eq(1, p_i) * rhs_eval`.
                    p_i * (rhs_eval - lhs_eval) + lhs_eval
                }
            }
        }

        let mle_evals = input
            .clone()
            .into_evals()
            .to_cpu()
            .into_iter()
            .map(|v| v.into())
            .collect::<Vec<_>>();

        eval(&mle_evals, point)
    }
}