#include "../include/blake2s.cuh"

__global__ void
commit_on_layer_aux(uint32_t size, uint32_t *column, char* result);

void commit_on_layer(uint32_t size, uint32_t *column, char* result) {
    unsigned int block_dim = 1024;

    unsigned int num_blocks = (size + block_dim - 1) / block_dim;
    commit_on_layer_aux<<<num_blocks, min(size, block_dim)>>>(
            size, column, result);
    cudaDeviceSynchronize();
}

__global__ void
commit_on_layer_aux(uint32_t size, uint32_t *column, char* result) {
    for(int i = 0; i < 32 * size; i++) {
        result[i] = 0;
    }

    //(0..(1 << log_size))
    //     .map(|i| {
    //         Blake2sMerkleHasher::hash_node(
    //             prev_layer.map(|prev_layer| (prev_layer[2 * i], prev_layer[2 * i + 1])),
    //             &columns.iter().map(|column| column.to_cpu()[i]).collect_vec(),
    //         )
    //     })
    //     .collect()
}
