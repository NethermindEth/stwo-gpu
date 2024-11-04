use crate::CudaBackend;
use num_traits::Zero;
use stwo_prover::core::{
    backend::Column, fields::{m31::BaseField, qm31::SecureField}, lookups::{
        gkr_prover::{correct_sum_as_poly_in_first_variable, EqEvals, GkrOps, Layer},
        mle::{Mle, MleOps}, sumcheck::MultivariatePolyOracle,
    }
};
use crate::cuda::{bindings, BaseFieldVec, SecureFieldVec};
use crate::cuda::bindings::CudaSecureField;

use self::bindings::CudaBaseField;

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

        if let Layer::GrandProduct(col) = layer {
            next_grand_product_layer(col)
        }
        else {
            println!("not supposed to access");
            layer.clone()
        }
        // match layer {
        //     Layer::GrandProduct(col) => next_grand_product_layer(col),
        //     Layer::LogUpGeneric {
        //         numerators,
        //         denominators,
        //     } => next_logup_generic_layer::<SecureField>(numerators, denominators),
        //     Layer::LogUpMultiplicities {
        //         numerators,
        //         denominators,
        //     } => next_logup_multiplicities_layer::<SecureField>(numerators, denominators),
        //     Layer::LogUpSingles { denominators } => next_logup_singles_layer::<SecureField>(denominators),
        // }
    }

    fn sum_as_poly_in_first_variable(
        h: &stwo_prover::core::lookups::gkr_prover::GkrMultivariatePolyOracle<'_, Self>,
        claim: SecureField,
    ) -> stwo_prover::core::lookups::utils::UnivariatePoly<SecureField> {
        // 1. Create each layer sum check 
        // 2. Polynomial interpolation check 

        let n_variables = h.n_variables();
        assert!(!n_variables.is_zero());
        let n_terms = 1 << (n_variables - 1);
        let eq_evals = h.eq_evals.as_ref();
        // Vector used to generate evaluations of `eq(x, y)` for `x` in the boolean hypercube.
        let y = eq_evals.y();
        let lambda = h.lambda;

        let (mut eval_at_0, mut eval_at_2) = {
            if let Layer::GrandProduct(col) = &h.input_layer {
                eval_grand_product_sum(eq_evals, col, n_terms)
            }
            else {
                println!("not supposed to access");
                (SecureField::default(), SecureField::default())
            }
        };
        
        // let (mut eval_at_0, mut eval_at_2) = match &h.input_layer {
        //     Layer::GrandProduct(col) => eval_grand_product_sum(eq_evals, col, n_terms),
        //     Layer::LogUpGeneric {
        //         numerators,
        //         denominators,
        //     } => eval_logup_sum(eq_evals, numerators, denominators, n_terms, lambda),
        //     Layer::LogUpMultiplicities {
        //         numerators,
        //         denominators,
        //     } => eval_logup_sum(eq_evals, numerators, denominators, n_terms, lambda),
        //     Layer::LogUpSingles { denominators } => {
        //         eval_logup_singles_sum(eq_evals, denominators, n_terms, lambda)
        //     }
        // };

        eval_at_0 *= h.eq_fixed_var_correction;
        eval_at_2 *= h.eq_fixed_var_correction;
        correct_sum_as_poly_in_first_variable(eval_at_0, eval_at_2, claim, y, n_variables)
    }
}

fn eval_grand_product_sum(
    eq_evals: &EqEvals<CudaBackend>,
    input_layer: &Mle<CudaBackend, SecureField>,
    n_terms: usize,
) -> (SecureField, SecureField) {
    // println!("n_terms: {}", n_terms); 
    // println!("eq_evals: {:?}", eq_evals);
    // println!("mle_evals: {:?}", &*eq_evals.clone().to_cpu());
    // println!("input_layer: {:?}", input_layer.clone().into_evals().to_cpu());
    let eval_at_0 = SecureFieldVec::new_uninitialized(1);
    let eval_at_2 = SecureFieldVec::new_uninitialized(1);

    unsafe {
        bindings::eval_grand_product_sum(
            eq_evals.device_ptr as *const CudaSecureField, 
            input_layer.device_ptr as *const CudaSecureField, 
            n_terms, 
            eval_at_0.device_ptr as *const CudaSecureField, 
            eval_at_2.device_ptr as *const CudaSecureField);
    };

    // println!("here: {:?}, {:?}\n", eval_at_0.to_cpu()[0].clone(), eval_at_2.to_cpu()[0].clone()); 
    (eval_at_0.to_cpu()[0].clone(), eval_at_2.to_cpu()[0].clone())
}


fn next_grand_product_layer(layer: &Mle<CudaBackend, SecureField>) -> Layer<CudaBackend> {
    let next_layer_size = layer.size / 2;
    let next_layer = SecureFieldVec::new_uninitialized(next_layer_size); 
    
    unsafe {
        bindings::next_grand_product_layer(
            layer.device_ptr as *const CudaSecureField, 
            layer.size, 
            next_layer.device_ptr as *const CudaSecureField, 
            next_layer_size);
    };
    
    Layer::GrandProduct(Mle::new(next_layer))
}

fn next_logup_generic_layer<F>(
    numerators: &Mle<CudaBackend, SecureField>,
    denominators: &Mle<CudaBackend, SecureField>,
) -> Layer<CudaBackend> {
    let next_layer_len = denominators.len() / 2; 
    let next_numerators = SecureFieldVec::new_uninitialized(next_layer_len);
    let next_denominators = SecureFieldVec::new_uninitialized(next_layer_len);

    unsafe {
        bindings::next_logup_generic_layer(
            numerators.device_ptr as *const CudaSecureField, 
            denominators.device_ptr as *const CudaSecureField, 
            numerators.size, 
            next_numerators.device_ptr as *const CudaSecureField, 
            next_denominators.device_ptr as *const CudaSecureField, 
            next_layer_len);
    };

    Layer::LogUpGeneric {
        numerators: Mle::new(next_numerators),
        denominators: Mle::new(next_denominators),
    }
}

fn next_logup_multiplicities_layer<F>(
    numerators: &Mle<CudaBackend, BaseField>,
    denominators: &Mle<CudaBackend, SecureField>,
) -> Layer<CudaBackend> {
    let next_layer_len = denominators.len() / 2; 
    let next_numerators = SecureFieldVec::new_uninitialized(next_layer_len);
    let next_denominators = SecureFieldVec::new_uninitialized(next_layer_len);

    unsafe {
        bindings::next_logup_multiplicities_layer(
            numerators.device_ptr as *const CudaBaseField, 
            denominators.device_ptr as *const CudaSecureField, 
            numerators.size, 
            next_numerators.device_ptr as *const CudaSecureField, 
            next_denominators.device_ptr as *const CudaSecureField, 
            next_layer_len);
    };

    Layer::LogUpGeneric {
        numerators: Mle::new(next_numerators),
        denominators: Mle::new(next_denominators),
    }
}

fn next_logup_singles_layer<F>(
    denominators: &Mle<CudaBackend, SecureField>,
) -> Layer<CudaBackend> {
    let next_layer_len = denominators.len() / 2; 
    let next_numerators = SecureFieldVec::new_uninitialized(next_layer_len);
    let next_denominators = SecureFieldVec::new_uninitialized(next_layer_len);

    unsafe {
        bindings::next_logup_singles_layer(
            denominators.device_ptr as *const CudaSecureField, 
            denominators.size, 
            next_numerators.device_ptr as *const CudaSecureField, 
            next_denominators.device_ptr as *const CudaSecureField, 
            next_layer_len);
    };

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

    #[test]
    fn grand_product_works() {
        const N: usize = 1 << 5;
        let values = Blake2sChannel::default().draw_felts(N);
        let product = values.iter().product();        

        let col_gpu = Mle::<CudaBackend, SecureField>::new(values.clone().into_iter().collect());
        let col_cpu = Mle::<CpuBackend, SecureField>::new(values.into_iter().collect());

        let input_layer = Layer::GrandProduct(col_gpu.clone());

        let (proof, _) = prove_batch(&mut Blake2sChannel::default(), vec![input_layer]);
        let GkrArtifact {
            ood_point,
            claims_to_verify_by_instance,
            n_variables_by_instance: _,
        } = partially_verify_batch(vec![Gate::GrandProduct], &proof, &mut Blake2sChannel::default()).unwrap();

        assert_eq!(proof.output_claims_by_instance, [vec![product]]);
        assert_eq!(
            claims_to_verify_by_instance,
            [vec![eval_at_point(&col_cpu, &ood_point)]]
        );
    }

    pub(crate) fn eval_at_point(input: &Mle<CpuBackend, SecureField>, point: &[SecureField]) -> SecureField {
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