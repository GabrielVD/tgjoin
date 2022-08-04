#include <ssjoin_verify.cuh>
#include <helper_cuda.h>
#include <helper_mem.cuh>

#define FILL ((overlap_t)-1)

__device__ static record_pair split_index(size_t overlap_index)
{
    record_pair pair;
    pair.id_high = tri_maxfit_d(overlap_index);
    pair.id_low = tri_rowstart(pair.id_high);
    
    while (pair.id_low > overlap_index)
    {
        pair.id_low = tri_rowstart(--pair.id_high);
    }
    pair.id_low = overlap_index - pair.id_low;
}

__global__ void verify(
    int* __restrict__ buffer_d,
    overlap_pack* __restrict__ overlap_pack_d,
    const size_t pack_count)
{
    extern __shared__ size_t shared[];
    int *batch_counter = (int*)shared;
    int *out_counter = (int*)(shared + 1);
    size_t *batch_buffer = shared + 2;
    record_pair *out_buffer = (record_pair*)(batch_buffer + blockDim.x);

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
                overlap_index = idx * OVERLAP_PACK_SIZE;
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
                    out_buffer[atomicAdd(out_counter, 1)] = pair;
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

    if (threadIdx.x == 0) { atomicAdd(buffer_d, candidates); }
}
