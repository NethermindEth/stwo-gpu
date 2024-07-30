#include "../include/blake2s.cuh"
#include <cstdio>

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

/* ************************* */

static __constant__ const unsigned int IV[8] = {
        0x6A09E667,
        0xBB67AE85,
        0x3C6EF372,
        0xA54FF53A,
        0x510E527F,
        0x9B05688C,
        0x1F83D9AB,
        0x5BE0CD19,
};

#define ROUND(r)                          \
    G(r, 0, v[0], v[4], v[8], v[12], m);  \
    G(r, 1, v[1], v[5], v[9], v[13], m);  \
    G(r, 2, v[2], v[6], v[10], v[14], m); \
    G(r, 3, v[3], v[7], v[11], v[15], m); \
    G(r, 4, v[0], v[5], v[10], v[15], m); \
    G(r, 5, v[1], v[6], v[11], v[12], m); \
    G(r, 6, v[2], v[7], v[8], v[13], m);  \
    G(r, 7, v[3], v[4], v[9], v[14], m);

__device__ __forceinline__ static unsigned int rotr(unsigned int x, unsigned int c) {
    return (x >> c) | (x << (32 - c));
}

static __constant__ const unsigned char SIGMA[12][16] = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
        {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
        {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
        {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
        {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
        {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
        {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
        {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
        {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
        {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
};

__device__ __forceinline__ static void G(const int r, const int i, unsigned int &a, unsigned int &b, unsigned int &c, unsigned int &d, unsigned int const m[16]) {
    a = a + b + m[SIGMA[r][2 * i]];
    d = rotr(d ^ a, 16);
    c = c + d;
    b = rotr(b ^ c, 12);
    a = a + b + m[SIGMA[r][2 * i + 1]];
    d = rotr(d ^ a, 8);
    c = c + d;
    b = rotr(b ^ c, 7);
}

__device__ void compress(H *state, unsigned int m[16]) {
    unsigned int v[16] = {
            state->s[0],
            state->s[1],
            state->s[2],
            state->s[3],
            state->s[4],
            state->s[5],
            state->s[6],
            state->s[7],
            IV[0],
            IV[1],
            IV[2],
            IV[3],
            IV[4],
            IV[5],
            IV[6],
            IV[7],
    };

    ROUND(0);
    ROUND(1);
    ROUND(2);
    ROUND(3);
    ROUND(4);
    ROUND(5);
    ROUND(6);
    ROUND(7);
    ROUND(8);
    ROUND(9);

    state->s[0] ^= v[0] ^ v[8];
    state->s[1] ^= v[1] ^ v[9];
    state->s[2] ^= v[2] ^ v[10];
    state->s[3] ^= v[3] ^ v[11];
    state->s[4] ^= v[4] ^ v[12];
    state->s[5] ^= v[5] ^ v[13];
    state->s[6] ^= v[6] ^ v[14];
    state->s[7] ^= v[7] ^ v[15];
}

__device__ void compress_cols(H *state, unsigned int **cols, int n_cols, unsigned int idx)
{
    int i;
    for (i = 0; i + 15 < n_cols; i += 16)
    {
        unsigned int msg[16] = {0};
        for (int j = 0; j < 16; j++)
        {
            msg[j] = cols[i + j][idx];
        }
        compress(state, msg);
    }

    if (i == n_cols)
    {
        return;
    }

    // Remainder.
    unsigned int msg[16] = {0};
    for (int j = 0; i < n_cols; i++, j++)
    {
        msg[j] = cols[i][idx];
    }
    compress(state, msg);
}

__global__ void blake_2s_hash_aux(uint32_t size, unsigned int *data, H *result) {
    unsigned int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx >= size)
        return;

    H state = {0};
    printf(
        "*********************************************************\n"
        "%d"
        "\n*********************************************************\n",
        data[0]
    );
    compress_cols(&state, &data, (int)size, idx);

    result[idx] = state;
}

void blake_2s_hash(uint32_t size, unsigned int *data, H *result) {
    unsigned int block_dim = 1024;

    unsigned int num_blocks = (size + block_dim - 1) / block_dim;


    blake_2s_hash_aux<<<num_blocks, min(size, block_dim)>>>(
            size, data, result
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError(); // Get the last error
    if (err != cudaSuccess) {
        // Print the error message
        printf("****************\n");
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        printf("****************\n");
    }
}
