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
        let size = 1 << log_size;
        let mut result = vec![Blake2sHash::default(); size];
        let result_pointer = result.as_mut_ptr();
        let device_column_pointers_vector: Vec<*const u32> = columns
            .iter()
            .map(|column| column.device_ptr)
            .collect();
        let number_of_columns = columns.len();

        unsafe{
            let device_result_pointer = bindings::copy_blake_2s_hash_vec_from_host_to_device(
                result_pointer, size
            );

            let device_column_pointers: *const *const u32 = bindings::copy_device_pointer_vec_from_host_to_device(
                device_column_pointers_vector.as_ptr(), number_of_columns
            );

            bindings::commit_on_first_layer(size, number_of_columns, device_column_pointers, device_result_pointer);

            bindings::copy_blake_2s_hash_vec_from_device_to_host(
                device_result_pointer, result_pointer, size,
            );
            bindings::free_blake_2s_hash_vec(device_result_pointer);
            bindings::free_device_pointer_vec(device_column_pointers);
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
    fn test_commit_on_first_layer_with_one_column_compared_with_cpu() {
        let log_size = 11;
        let size = 1 << log_size;

        let cpu_column = vec![M31::from(1)].repeat(size);
        let cpu_columns_vector: Vec<&Vec<BaseField>> = vec![&cpu_column];
        let cpu_columns: &[&Vec<BaseField>] = cpu_columns_vector.as_slice();

        let gpu_column = cpu_column.clone();
        let base_field_vector: BaseFieldVec = BaseFieldVec::from_vec(gpu_column);
        let gpu_columns_vector: Vec<&BaseFieldVec> = vec![&base_field_vector];
        let gpu_columns = gpu_columns_vector.as_slice();

        let expected_result = <CpuBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(log_size, None, cpu_columns);
        let result = CudaBackend::commit_on_layer(log_size, None, gpu_columns);

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_commit_on_first_layer_with_many_columns_compared_with_cpu() {
        let log_size = 11;
        let size = 1 << log_size;

        let cpu_columns_vector: Vec<Vec<BaseField>> = (0..35).map(|index|
            vec![M31::from(index)].repeat(size)
        ).collect();

        let columns = cpu_columns_vector.clone();
        let gpu_columns_vector: Vec<BaseFieldVec> = columns.into_iter().map(|vector|
            BaseFieldVec::from_vec(vector)
        ).collect();

        let expected_result = <CpuBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(log_size, None, &cpu_columns_vector.iter().collect::<Vec<_>>());
        let result = CudaBackend::commit_on_layer(log_size, None, &gpu_columns_vector.iter().collect::<Vec<_>>());

        assert_eq!(result, expected_result);
    }
}
