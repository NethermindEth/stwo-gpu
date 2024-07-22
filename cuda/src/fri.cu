#include "../include/fri.cuh"

__global__ void sum_reduce(uint32_t *list, uint32_t *temp_list, uint32_t *results, const uint32_t list_size) {
    const uint32_t block_thread_index = threadIdx.x;
    const uint32_t first_thread_in_block_index = blockIdx.x * blockDim.x;
    const uint32_t grid_thread_index = first_thread_in_block_index + block_thread_index;
    const uint32_t half_list_size = list_size >> 1;

    if (grid_thread_index < half_list_size) {
        uint32_t *list_to_sum_in_block = &temp_list[first_thread_in_block_index];
        uint32_t &thread_result = list_to_sum_in_block[block_thread_index];

        thread_result = sub(
                list[grid_thread_index],
                list[grid_thread_index + half_list_size]);

        __syncthreads();

        uint32_t list_to_sum_in_block_half_size = min(half_list_size, blockDim.x) >> 1;
        while (block_thread_index < list_to_sum_in_block_half_size) {
            thread_result = add(
                    thread_result, list_to_sum_in_block[block_thread_index + list_to_sum_in_block_half_size]);

            __syncthreads();

            list_to_sum_in_block_half_size >>= 1;
        }

        const bool is_first_thread_in_block = block_thread_index == 0;
        if (is_first_thread_in_block) {
            results[blockIdx.x] = thread_result;
        }
    }
}

extern "C"
void sum(uint32_t *list, uint32_t *temp_list, uint32_t *results, const uint32_t list_size) {
    sum_reduce<<<1, 512>>>(list, temp_list, results, list_size);
}