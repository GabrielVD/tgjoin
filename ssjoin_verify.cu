#include <ssjoin_verify.cuh>
#include <helper_cuda.h>
#include <helper_mem.cuh>

#define FILL ((overlap_t)-1)

__device__ static record_pair split_index(size_t overlap_index)
{
    size_t n = tri_maxfit_d(overlap_index);
    size_t nsum = tri_rowstart(n);
    
    while (nsum > overlap_index) { nsum = tri_rowstart(--n); }

    record_pair pair;
    pair.id_high = n;
    pair.id_low = overlap_index - nsum;
    return pair;
}

__global__ void verify(
    const record_t* __restrict__ record_map_d,
    record_pair* __restrict__ out_d,
    int* __restrict__ out_count_d,
    int* __restrict__ candidates_d,
    overlap_pack* __restrict__ overlap_pack_d,
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
                    pair.id_low = record_map_d[record_map_d[pair.id_low] - 2];
                    pair.id_high = record_map_d[record_map_d[pair.id_high] - 2];
                    overlap_t overlap = overlap_buffer[threadIdx.x];
                    if (threadIdx.x == 0) { out_buffer[atomicAdd(out_counter, 1)] = pair; }
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
