use stwo_prover::core::{
    backend::Column,
    fields::{m31::BaseField, qm31::SecureField},
    lookups::mle::{Mle, MleOps},
};

use crate::{
    cuda::{self, BaseFieldVec, SecureFieldVec},
    CudaBackend,
};

impl MleOps<BaseField> for CudaBackend {
    fn fix_first_variable(
        mle: Mle<Self, BaseField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField>
    where
        Self: MleOps<SecureField>,
    {   
        let evals_size = mle.len();
        let result_evals = SecureFieldVec::new_uninitialized(evals_size >> 1);
        unsafe {
            cuda::bindings::fix_first_variable_base_field(
                mle.into_evals().device_ptr,
                evals_size,
                assignment.into(),
                result_evals.device_ptr,
            )
        }
        Mle::new(result_evals)
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
        let evals_size = mle.len();
        let result_evals = SecureFieldVec::new_uninitialized(evals_size >> 1);
        unsafe {
            cuda::bindings::fix_first_variable_secure_field(
                mle.into_evals().device_ptr,
                evals_size,
                assignment.into(),
                result_evals.device_ptr,
            )
        }

        Mle::new(result_evals)
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use stwo_prover::core::{
        backend::{Column, CpuBackend},
        fields::{m31::BaseField, qm31::SecureField},
        lookups::mle::{Mle, MleOps},
    };

    use crate::{
        cuda::{BaseFieldVec, SecureFieldVec},
        CudaBackend,
    };

    #[test]
    fn fix_first_variable_with_base_field_mle_matches_cpu() {
        const N_VARIABLES: u32 = 8;

        let values = (0..1 << N_VARIABLES).map(BaseField::from).collect_vec();

        let mle_cuda = Mle::<CudaBackend, BaseField>::new(BaseFieldVec::from_vec(values.clone()));
        let mle_cpu = Mle::<CpuBackend, BaseField>::new(values);
        let random_assignment = SecureField::from_u32_unchecked(7, 12, 3, 2);
        let mle_fixed_cpu = MleOps::<BaseField>::fix_first_variable(mle_cpu, random_assignment);
        
        let mle_fixed_simd = MleOps::<BaseField>::fix_first_variable(mle_cuda, random_assignment);

        assert_eq!(mle_fixed_simd.into_evals().to_cpu(), *mle_fixed_cpu)
    }

    #[test]
    fn fix_first_variable_with_secure_field_mle_matches_cpu() {
        const N_VARIABLES: u32 = 8;

        let values = (0..(1 << N_VARIABLES) * 4)
            .collect_vec()
            .chunks(4)
            .map(|a| SecureField::from_u32_unchecked(a[0], a[1], a[2], a[3]))
            .collect_vec();

        let mle_cuda =
            Mle::<CudaBackend, SecureField>::new(SecureFieldVec::from_vec(values.clone()));
        let mle_cpu = Mle::<CpuBackend, SecureField>::new(values);
        let random_assignment = SecureField::from_u32_unchecked(7, 12, 3, 2);
        let mle_fixed_cpu = MleOps::<SecureField>::fix_first_variable(mle_cpu, random_assignment);

        let mle_fixed_simd = MleOps::<SecureField>::fix_first_variable(mle_cuda, random_assignment);

        assert_eq!(mle_fixed_simd.into_evals().to_cpu(), *mle_fixed_cpu)
    }

    
    #[test]
    fn fix_first_variable_with_secure_field_mle_matches_cpu_test() {
        const N_VARIABLES: u32 = 8;

        let values = (0..(1 << N_VARIABLES) * 4)
            .collect_vec()
            .chunks(4)
            .map(|a| SecureField::from_u32_unchecked(a[0], a[1], a[2], a[3]))
            .collect_vec();
        let values = vec![
            SecureField::from_u32_unchecked(582331529, 613033660, 368449590, 1538560393),
            SecureField::from_u32_unchecked(251659788, 78954062, 1801023691, 998786282),
            SecureField::from_u32_unchecked(1184369939, 1548123522, 496550933, 2088405710),
            SecureField::from_u32_unchecked(1158835792, 413758273, 961347360, 1684210072),
        ];

        let mle_cuda =
            Mle::<CudaBackend, SecureField>::new(SecureFieldVec::from_vec(values.clone()));
        let mle_cpu = Mle::<CpuBackend, SecureField>::new(values);
        let random_assignment = SecureField::from_u32_unchecked(7, 12, 3, 2);
        let mle_fixed_cpu = MleOps::<SecureField>::fix_first_variable(mle_cpu, random_assignment);

        let mle_fixed_simd = MleOps::<SecureField>::fix_first_variable(mle_cuda, random_assignment);

        assert_eq!(mle_fixed_simd.into_evals().to_cpu(), *mle_fixed_cpu)
    }
}
