use itertools::Itertools;
use stwo_prover::core::backend::{Column, ColumnOps};
use stwo_prover::core::vcs::blake2_hash::Blake2sHash;
use stwo_prover::core::vcs::blake2_merkle::Blake2sMerkleHasher;
use stwo_prover::core::vcs::ops::{MerkleHasher, MerkleOps};

use crate::cuda::{BaseFieldVec, bindings};
use crate::CudaBackend;

impl ColumnOps<Blake2sHash> for CudaBackend {
    type Column = Vec<Blake2sHash>;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}


impl MerkleOps<Blake2sMerkleHasher> for CudaBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Vec<Blake2sHash>>,
        columns: &[&BaseFieldVec],
    ) -> Vec<Blake2sHash> {
        let column = columns[0];
        let size = 1 << log_size;
        let mut result = vec![Blake2sHash::default(); size];
        let result_pointer = result.as_mut_ptr();

        unsafe{
            let device_result_pointer = bindings::copy_blake_2s_hash_vec_from_host_to_device(
                result_pointer, size
            );

            bindings::commit_on_first_layer(size, column.device_ptr, device_result_pointer);

            bindings::copy_blake_2s_hash_vec_from_device_to_host(
                device_result_pointer, result_pointer, size,
            );
            bindings::free_blake_2s_hash_vec(device_result_pointer);
        }

        return result.to_vec();
    }
}

#[cfg(test)]
mod tests {
    use stwo_prover::core::backend::CpuBackend;
    use stwo_prover::core::fields::m31::{BaseField, M31};
    use stwo_prover::core::vcs::blake2_merkle::Blake2sMerkleHasher;
    use stwo_prover::core::vcs::ops::MerkleOps;

    use crate::cuda::BaseFieldVec;
    use crate::CudaBackend;

    #[test]
    fn test_commit_on_first_layer_compared_with_cpu() {
        let log_size = 11;
        let size = 1 << log_size;

        let cpu_column = vec![M31::from(1)].repeat(size);
        let columns: Vec<&Vec<BaseField>> = vec![&cpu_column];
        let cpu_columns: &[&Vec<BaseField>] = columns.as_slice();

        let gpu_column = cpu_column.clone();
        let base_field_vector: BaseFieldVec = BaseFieldVec::from_vec(gpu_column);
        let gpu_columns_vector: Vec<&BaseFieldVec> = vec![&base_field_vector];
        let gpu_columns = gpu_columns_vector.as_slice();

        let expected_result = <CpuBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(log_size, None, cpu_columns);
        let result = CudaBackend::commit_on_layer(log_size, None, gpu_columns);

        assert_eq!(result, expected_result);
    }
}
