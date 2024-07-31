use stwo_prover::core::backend::ColumnOps;
use stwo_prover::core::vcs::blake2_hash::Blake2sHash;
use stwo_prover::core::vcs::blake2_merkle::Blake2sMerkleHasher;
use stwo_prover::core::vcs::ops::MerkleOps;

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
        let number_of_columns = columns.len();

        let result = vec![Blake2sHash::default(); size];
        let result_pointer = result.as_ptr();

        unsafe {
            Self::commit_on_layer_using_gpu(size, number_of_columns, columns, prev_layer, result_pointer);
        }

        return result.to_vec();
    }
}

impl CudaBackend {
    unsafe fn commit_on_layer_using_gpu(size: usize, number_of_columns: usize, columns: &[&BaseFieldVec], prev_layer: Option<&Vec<Blake2sHash>>, result_pointer: *const Blake2sHash) {
        let device_column_pointers_vector: Vec<*const u32> = columns
            .iter()
            .map(|column| column.device_ptr)
            .collect();

        let device_result_pointer = bindings::copy_blake_2s_hash_vec_from_host_to_device(
            result_pointer, size
        );

        let device_column_pointers: *const *const u32 = bindings::copy_device_pointer_vec_from_host_to_device(
            device_column_pointers_vector.as_ptr(), number_of_columns
        );

        if let Some(previous_layer) = prev_layer {
            let device_previous_layer_pointer = bindings::copy_blake_2s_hash_vec_from_host_to_device(
                previous_layer.as_ptr(), size << 1
            );

            bindings::commit_on_layer_with_previous(
                size, number_of_columns, device_column_pointers, device_previous_layer_pointer, device_result_pointer
            );

            bindings::free_blake_2s_hash_vec(device_previous_layer_pointer);
        } else {
            bindings::commit_on_first_layer(
                size, number_of_columns, device_column_pointers, device_result_pointer
            );
        }

        bindings::copy_blake_2s_hash_vec_from_device_to_host(
            device_result_pointer, result_pointer, size,
        );
        bindings::free_blake_2s_hash_vec(device_result_pointer);
        bindings::free_device_pointer_vec(device_column_pointers);
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
    fn test_commit_on_first_layer_with_many_columns_compared_with_cpu() {
        let log_size = 11;
        let size = 1 << log_size;

        let cpu_columns_vector: Vec<Vec<BaseField>> = columns_test_vector(35, size);
        let gpu_columns_vector: Vec<BaseFieldVec> = gpu_columns_from(&cpu_columns_vector);

        let expected_result = <CpuBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(log_size, None, &cpu_columns_vector.iter().collect::<Vec<_>>());
        let result = CudaBackend::commit_on_layer(log_size, None, &gpu_columns_vector.iter().collect::<Vec<_>>());

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_commit_on_layer_with_previous_layer_compared_with_cpu() {
        let current_layer_log_size = 11;
        let current_layer_size = 1 << current_layer_log_size;
        let previous_layer_log_size = current_layer_log_size + 1;
        let previous_layer_size = 1 << previous_layer_log_size;
        let no_previous_layer = None;

        // First layer

        let cpu_columns_vector: Vec<Vec<BaseField>> = columns_test_vector(35, previous_layer_size);
        let gpu_columns_vector: Vec<BaseFieldVec> = gpu_columns_from(&cpu_columns_vector);

        let cpu_previous_layer = <CpuBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(previous_layer_log_size, no_previous_layer, &cpu_columns_vector.iter().collect::<Vec<_>>());
        let gpu_previous_layer = CudaBackend::commit_on_layer(previous_layer_log_size, no_previous_layer, &gpu_columns_vector.iter().collect::<Vec<_>>());

        // Current layer

        let cpu_columns_vector: Vec<Vec<BaseField>> = columns_test_vector(16, current_layer_size);
        let gpu_columns_vector: Vec<BaseFieldVec> = gpu_columns_from(&cpu_columns_vector);

        let expected_result = <CpuBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(current_layer_log_size, Some(&cpu_previous_layer), &cpu_columns_vector.iter().collect::<Vec<_>>());
        let result = CudaBackend::commit_on_layer(current_layer_log_size, Some(&gpu_previous_layer), &gpu_columns_vector.iter().collect::<Vec<_>>());

        assert_eq!(result, expected_result);
    }

    fn gpu_columns_from(columns: &Vec<Vec<BaseField>>) -> Vec<BaseFieldVec> {
        columns.clone().into_iter().map(|vector|
            BaseFieldVec::from_vec(vector)
        ).collect()
    }

    fn columns_test_vector(number_of_columns: usize, size_of_columns: usize) -> Vec<Vec<BaseField>> {
        (0..number_of_columns).map(|index_of_column|
            (0..size_of_columns).map(|index_in_column|
                M31::from(index_in_column * index_of_column)
            ).collect()
        ).collect()
    }
}
