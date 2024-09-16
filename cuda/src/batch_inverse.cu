#include "batch_inverse.cuh"
#include "utils.cuh"

template<typename T>
__device__ void new_forward_parent(T *from, T *dst, int index) {
    // Computes the value of the parent from the multiplication of two children.
    // dst  : Pointer to the beginning of the parent's level.
    // from : Pointer to the beginning of the children level.
    // index: Index of the computed parent.
    dst[index] = mul(from[index << 1], from[(index << 1) + 1]);
}

template<typename T>
__device__ void new_backward_children(T *from, T *dst, int index) {
    // Computes the inverse of the two children from the inverse of the parent.
    // dst  : Pointer to the beginning of the children's level.
    // from : Pointer to the beginning of the parent's level.
    // index: Index of the computed children.
    T temp = dst[index << 1];
    dst[index << 1] = mul(from[index], dst[(index << 1) + 1]);
    dst[(index << 1) + 1] = mul(from[index], temp);
}

template<typename T>
__device__ void batch_inverse(T *from, T *dst, int size, int log_size, T *s_from, T *s_inner_tree) {
    // Input:
    // - from      : array of T representing field elements.
    // - inner_tree: array of T used as an auxiliary variable.
    // - size      : size of "from" and "inner_tree".
    // - log_size  : log(size).
    // Output:
    // - dst       : array of T with the inverses of "from".
    //
    // Variation of Montgomery's trick to leverage GPU parallelization.
    // Construct a binary tree:
    //    - from      : stores the leaves
    //    - inner_tree: stores the inner nodes and the root.
    // 
    // The algorithm has three parts:
    //    - Cumulative product: each parent is the product of its children.
    //    - Compute inverse of root node
    //    - Backward pass: compute inverses of children using the fact that
    //          inv_left_child  = inv_parent * right_child
    //          inv_right_child = inv_parent * left_child
    int index = threadIdx.x;

    s_from[index] = from[2 * blockIdx.x * blockDim.x + index];
    s_from[index + blockDim.x] = from[2 * blockIdx.x * blockDim.x + index + blockDim.x];
    __syncthreads();

    dst = &dst[2 * blockIdx.x * blockDim.x];

    // Size tracks the number of threads working.

    size = size >> 1;

    // The first level is a special case because inner_tree and leaves
    // are stored in separate variables.
    if(index < size) {
        new_forward_parent(s_from, s_inner_tree, index);
        // from      : | a_0       | a_1       | ... | a_(n/2 - 1)       |      ...    | a_(n-1)
        // inner_tree: | a_0 * a_1 | a_2 * a_3 | ... | a_(n-2) * a_(n-1) | empty | ... | empty   
    }

    int from_offset = 0;   // Offset at inner_tree to get the children.
    int dst_offset = size; // Offset at inner_tree to store the parents.
    size >>= 1;            // Next level is half the size.

    // Each step will compute one level of the inner_tree.
    // If size = 4 inner tree stores:
    // |       Level 1         |        Root           |
    // | a_0 * a_1 | a_2 * a_3 | a_0 * a_1 * a_2 * a_3 |
    // Construct tree up to the level with 32 leaves to leverage
    // SIMD synchronization within a warp
    int step = 1;
    while(step + 5 < log_size) {
        __syncthreads();

        if(index < size) {
            // Each thread computes one parent as the product of left and right children
            new_forward_parent(&s_inner_tree[from_offset], &s_inner_tree[dst_offset], index);
        }

        from_offset = dst_offset;       // Output of this level is input of next one.
        dst_offset = dst_offset + size; // Skip the number of nodes computed.

        size >>= 1; // Next level is half the size.
        step++;
    }
    
    // Compute inverse of the root.
    __syncthreads();
    if(index < (size << 1)){
        s_inner_tree[from_offset + index] = inv(s_inner_tree[from_offset + index]);
    }
    
    // Backward Pass: compute the inverses of the children using the parents.
    step = 5;
    size = 32;
    //from_offset = dst_offset - 1;
    dst_offset = from_offset - (size << 1);
    while(step < log_size - 1) {
        __syncthreads();
        if(index < size) {
            // Compute children inverses from parent inverses.
            new_backward_children(&s_inner_tree[from_offset], &s_inner_tree[dst_offset], index);
        }

        size <<= 1; // Each level doubles up its size.

        from_offset = dst_offset;               // Output of this level is input of next one.
        dst_offset = from_offset - (size << 1); // Size threads work but 2*size children are computed.

        step++;
    }
    
    __syncthreads();
    // The inner_tree has all its inverses computed, now
    // we have to compute the inverses of the leaves:
    
    if(index < size) {
        dst[index << 1] = mul(s_inner_tree[index], s_from[(index << 1) + 1]);
        dst[(index << 1) + 1] = mul(s_inner_tree[index], s_from[index << 1]);
    }
}

__global__ void batch_inverse_base_field_kernel(m31 *from, m31 *dst, int size, int log_size) {
    // Thread syncing happens within a block. 
    // Split the problem to feed them to multiple blocks.
    if(size >= 512) {
        size = 512;
        log_size = 9;
    }

    extern __shared__ m31 shared_basefield[];
    m31 *s_from_basefield = shared_basefield;
    m31 *s_inner_trees_basefield = &shared_basefield[size];

    batch_inverse(from, dst, size, log_size, s_from_basefield, s_inner_trees_basefield);
}

__global__ void batch_inverse_secure_field_kernel(qm31 *from, qm31 *dst, int size, int log_size) {
    // Thread syncing happens within a block. 
    // Split the problem to feed them to multiple blocks.
    if(size >= 1024) {
        size = 1024;
        log_size = 10;
    }

    extern __shared__ qm31 shared_qm31[];
    qm31 *s_from_qm31 = shared_qm31;
    qm31 *s_inner_trees_qm31 = &shared_qm31[size];

    batch_inverse(from, dst, size, log_size, s_from_qm31, s_inner_trees_qm31);
}

void batch_inverse_base_field(m31 *from, m31 *dst, int size) {
    int log_size = log_2(size);
    int block_size = 256;
    int half_size = size >> 1;
    int num_blocks = (half_size + block_size - 1) / block_size;
    int shared_memory_bytes = 512 * 4  + (512 - 32) * 4;

    batch_inverse_base_field_kernel<<<num_blocks, block_size, shared_memory_bytes>>>(from, dst, size, log_size);
    cudaDeviceSynchronize();
}

void batch_inverse_secure_field(qm31 *from, qm31 *dst, int size) {
    int log_size = log_2(size);
    int block_size = 512;
    int half_size = size >> 1;
    int num_blocks = (half_size + block_size - 1) / block_size;
    int shared_memory_bytes = 1024 * 4 * 4 + (1024 - 32) * 4 * 4;

    batch_inverse_secure_field_kernel<<<num_blocks, block_size, shared_memory_bytes>>>(from, dst, size, log_size);
    cudaDeviceSynchronize();
}