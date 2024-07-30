use itertools::Itertools;
use stwo_prover::core::backend::{Col, Column, ColumnOps};
use stwo_prover::core::fields::m31::M31;
use stwo_prover::core::vcs::blake2_hash::{Blake2sHash, Blake2sHasher};
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
        let result_vector = vec![Blake2sHash::default(); size];
        let result = result_vector.as_slice();

        // unsafe{
        //     bindings::commit_on_first_layer(
        //         size,
        //         column.device_ptr,
        //         result,
        //     )
        // }

        return result.to_vec();
    }
}

fn blake_2s_hash_gpu(size: usize, data: &Vec<M31>) -> Blake2sHash {
    let mut result: Blake2sHash = Default::default();
    let result_pointer: *mut Blake2sHash = &mut result;

    unsafe {
        let device_data = bindings::copy_uint32_t_vec_from_host_to_device(
            data.as_ptr() as *const u32,
            data.len() as u32,
        );
        let device_result_pointer = bindings::copy_blake_2s_hash_from_host_to_device(
            result_pointer
        );

        bindings::blake_2s_hash(size, device_data, device_result_pointer);

        bindings::copy_blake_2s_hash_from_device_to_host(
            device_result_pointer,
            result_pointer,
        );

        bindings::free_uint32_t_vec(device_data);
        bindings::free_blake_2s_hash(result_pointer);
    }
    return result;
}

#[cfg(test)]
mod tests {
    use stwo_prover::core::backend::CpuBackend;
    use stwo_prover::core::fields::m31::{BaseField, M31};
    use stwo_prover::core::vcs::blake2_hash::Blake2sHasher;
    use stwo_prover::core::vcs::blake2_merkle::Blake2sMerkleHasher;
    use stwo_prover::core::vcs::hasher::Hasher;
    use stwo_prover::core::vcs::ops::{MerkleHasher, MerkleOps};
    use crate::blake2s::blake_2s_hash_gpu;
    use crate::cuda::BaseFieldVec;
    use crate::CudaBackend;

    #[test]
    fn test_blake_2s_hash() {
        const DATA_SIZE: usize = 4096;
        let data = vec![M31::from(12345); DATA_SIZE];

        let expected_result = Blake2sMerkleHasher::hash_node(None, &data);
        let result = blake_2s_hash_gpu(data.len(), &data);

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_commit_on_first_layer_compared_with_cpu() {
        let log_size = 3;
        let size = 1 << log_size;

        let column = vec![M31::from(1)].repeat(size);

        let columns: Vec<&Vec<BaseField>> = vec![&column];
        let columns_for_cpu: &[&Vec<BaseField>] = columns.as_slice();

        let base_field_vector: BaseFieldVec = BaseFieldVec::from_vec(column.clone());
        let columns_for_gpu: Vec<&BaseFieldVec> = vec![&base_field_vector];

        let expected_result = <CpuBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(log_size, None, columns_for_cpu);
        let result = CudaBackend::commit_on_layer(log_size, None, columns_for_gpu.as_slice());

        assert_eq!(result, expected_result);
    }
}
