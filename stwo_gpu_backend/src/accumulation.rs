use stwo_prover::core::{
    air::accumulation::AccumulationOps, fields::secure_column::SecureColumnByCoords,
};

use crate::backend::CudaBackend;
use crate::cuda::bindings;
use crate::secure_column::CudaSecureColumn;

impl AccumulationOps for CudaBackend {
    fn accumulate(column: &mut SecureColumnByCoords<Self>, other: &SecureColumnByCoords<Self>) {
        unsafe {
            bindings::accumulate(
                column.len() as u32,
                CudaSecureColumn::from(column).device_ptr(),
                CudaSecureColumn::from(other).device_ptr(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use stwo_prover::core::air::accumulation::AccumulationOps;
    use stwo_prover::core::backend::Column;
    use stwo_prover::core::fields::m31::{M31, P};
    use stwo_prover::core::fields::secure_column::SecureColumnByCoords;

    use crate::backend::CudaBackend;
    use crate::cuda::BaseFieldVec;

    #[test]
    fn test_accumulation() {
        let size = 2 << 20;
        let left_summand: [BaseFieldVec; 4] = [
            BaseFieldVec::from_vec(vec![M31::from(1)].repeat(size)),
            BaseFieldVec::from_vec(vec![M31::from(2)].repeat(size)),
            BaseFieldVec::from_vec(vec![M31::from(3)].repeat(size)),
            BaseFieldVec::from_vec(vec![M31::from(4)].repeat(size)),
        ];
        let right_summand: [BaseFieldVec; 4] = [
            BaseFieldVec::from_vec(vec![M31::from(5)].repeat(size)),
            BaseFieldVec::from_vec(vec![M31::from(6)].repeat(size)),
            BaseFieldVec::from_vec(vec![M31::from(7)].repeat(size)),
            BaseFieldVec::from_vec(vec![M31::from(8)].repeat(size)),
        ];

        let mut left_secure_column = SecureColumnByCoords {
            columns: left_summand,
        };
        let right_secure_column = SecureColumnByCoords {
            columns: right_summand,
        };
        CudaBackend::accumulate(&mut left_secure_column, &right_secure_column);

        let expected_result = [
            vec![M31::from(6)].repeat(size),
            vec![M31::from(8)].repeat(size),
            vec![M31::from(10)].repeat(size),
            vec![M31::from(12)].repeat(size),
        ];
        assert_eq!(expected_result[0], left_secure_column.columns[0].to_cpu());
        assert_eq!(expected_result[1], left_secure_column.columns[1].to_cpu());
        assert_eq!(expected_result[2], left_secure_column.columns[2].to_cpu());
        assert_eq!(expected_result[3], left_secure_column.columns[3].to_cpu());
    }

    #[test]
    fn test_accumulation_m31_arithmetic() {
        let size = 2 << 20;
        let left_summand: [BaseFieldVec; 4] = [
            BaseFieldVec::from_vec(vec![M31::from(1)].repeat(size)),
            BaseFieldVec::from_vec(vec![M31::from(1)].repeat(size)),
            BaseFieldVec::from_vec(vec![M31::from(1)].repeat(size)),
            BaseFieldVec::from_vec(vec![M31::from(1)].repeat(size)),
        ];
        let right_summand: [BaseFieldVec; 4] = [
            BaseFieldVec::from_vec(vec![M31::from(P - 1)].repeat(size)),
            BaseFieldVec::from_vec(vec![M31::from(P - 1)].repeat(size)),
            BaseFieldVec::from_vec(vec![M31::from(P - 1)].repeat(size)),
            BaseFieldVec::from_vec(vec![M31::from(P - 1)].repeat(size)),
        ];

        let mut left_secure_column = SecureColumnByCoords {
            columns: left_summand,
        };
        let right_secure_column = SecureColumnByCoords {
            columns: right_summand,
        };
        CudaBackend::accumulate(&mut left_secure_column, &right_secure_column);

        let expected_result = [
            vec![M31::from(0)].repeat(size),
            vec![M31::from(0)].repeat(size),
            vec![M31::from(0)].repeat(size),
            vec![M31::from(0)].repeat(size),
        ];
        assert_eq!(expected_result[0], left_secure_column.columns[0].to_cpu());
        assert_eq!(expected_result[1], left_secure_column.columns[1].to_cpu());
        assert_eq!(expected_result[2], left_secure_column.columns[2].to_cpu());
        assert_eq!(expected_result[3], left_secure_column.columns[3].to_cpu());
    }
}
