#include <ssjoin_verify.cuh>
#include <helper_cuda.h>
#include <helper_mem.cuh>
#include <similarity.cuh>

#define FILL ((overlap_t)-1)

__device__ __forceinline__ static record_pair split_index(size_t overlap_index)
{
    size_t n = tri_maxfit_d(overlap_index);
    size_t nsum = tri_rowstart(n);
    
    while (nsum > overlap_index) { nsum = tri_rowstart(--n); }

    record_pair pair;
    pair.id_high = n;
    pair.id_low = overlap_index - nsum;
    return pair;
}

__device__ __forceinline__ static bool are_sim(
    record_pair pair,
    overlap_t overlap,
    float threshold,
    float overlap_factor,
    const record_t *record_map_d)
{
    record_t head_low = record_map_d[pair.id_low];
    record_t limit_low = record_map_d[pair.id_low + 1] - 2;

    record_t head_high = record_map_d[pair.id_high];
    record_t limit_high = record_map_d[pair.id_high + 1] - 2;

    int overlap_needed;
    {
        auto size_low = limit_low - head_low;
        auto size_high = limit_high - head_high;
        if (overlap != FILL - 1)
        {
            head_low += index_prefix_size_d(size_low, overlap_factor);
            head_high += prefix_size_d(size_high, threshold);
        }
        else { overlap = 0; }
        overlap_needed = OVERLAP_D(size_low, size_high, overlap_factor) - overlap;
    }

    while (overlap_needed != 0
        && head_low != limit_low
        && head_high != limit_high)
    {
        auto token_low = record_map_d[head_low];
        auto token_high = record_map_d[head_high];
        overlap_needed -= token_low == token_high;
        head_low += token_low <= token_high;
        head_high += token_high <= token_low;
    }

    return overlap_needed == 0;
}

__global__ void verify(
    const record_t* __restrict__ record_map_d,
    record_pair* __restrict__ out_d,
    int* __restrict__ out_count_d,
    int* __restrict__ candidates_d,
    overlap_pack* __restrict__ overlap_pack_d,
    const float threshold,
    const float overlap_factor,
    const size_t overlap_offset,
    const size_t pack_count)
{
    extern __shared__ size_t shared[];
    int *batch_counter = (int*)shared;
    int *out_counter = (int*)(shared + 1);
    size_t *batch_buffer = shared + 2;
    record_pair *out_buffer = (record_pair*)(batch_buffer + blockDim.x);
    overlap_t *overlap_buffer = (overlap_t*)(out_buffer + blockDim.x);

    int candidates = 0;
    const size_t stride = STRIDE();

    if (threadIdx.x == 0)
    {
        *batch_counter = 0;
        *out_counter = 0;
    }
    __syncthreads();

    for (size_t block = blockIdx.x * blockDim.x; block < pack_count; block += stride)
    {
        // exchange a pack with zeroed memory
        size_t overlap_index;
        overlap_pack pack = 0;
        {
            size_t idx = block + threadIdx.x;
            if (idx < pack_count)
            {
                overlap_index = overlap_offset + idx * OVERLAP_PACK_SIZE;
                pack = atomicExch(overlap_pack_d + idx, 0);
            }
        }

        // fully process the packs read
        while (1)
        {
            // dump pack to shared memory until empty or batch is full
            while (pack != 0)
            {
                overlap_t overlap = pack & FILL;
                if (overlap != 0 && overlap != FILL)
                {
                    int old_count = atomicAdd(batch_counter, 1);
                    if (old_count >= blockDim.x) { break; }
                    batch_buffer[old_count] = overlap_index;
                    overlap_buffer[old_count] = overlap;
                }
                pack = pack >> (sizeof(overlap_t) * CHAR_BIT);
                ++overlap_index;
            }

            __syncthreads();
            int batch = *batch_counter;
            __syncthreads();

            // read next packs if batch is not full and not the last one
            if (batch < blockDim.x && (block + stride < pack_count || batch == 0))
            {
                break;
            }

            // process current batch
            {
                if (threadIdx.x < batch)
                {
                    record_pair pair = split_index(batch_buffer[threadIdx.x]);
                    overlap_t overlap = overlap_buffer[threadIdx.x];
                    if (are_sim(pair, overlap, threshold, overlap_factor, record_map_d))
                    {
                        pair.id_low = record_map_d[record_map_d[pair.id_low] - 2];
                        pair.id_high = record_map_d[record_map_d[pair.id_high] - 2];
                        out_buffer[atomicAdd(out_counter, 1)] = pair;
                    }
                }
            }

            {
                __syncthreads();
                int counter = *out_counter;
                // reserve output region
                if (threadIdx.x == 0)
                {
                    *batch_counter = atomicAdd(out_count_d, counter);
                }
                __syncthreads();
                int out_start = *batch_counter;
                // output pairs
                if (threadIdx.x < counter)
                {
                    out_d[out_start + threadIdx.x] = out_buffer[threadIdx.x];
                }
            }

            __syncthreads();
            if (threadIdx.x == 0)
            {
                candidates += blockDim.x < batch ? blockDim.x : batch;
                *batch_counter = 0;
                *out_counter = 0;
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0) { atomicAdd(candidates_d, candidates); }
}
