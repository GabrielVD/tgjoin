#include <ssjoin_verify.cuh>
#include <helper_cuda.h>

#define FILL ((overlap_t)-1)

__global__ void verify(
    int* __restrict__ buffer_d,
    overlap_pack* __restrict__ overlap_pack_d,
    const size_t pack_count)
{
    // count of elements in buffer
    extern __shared__ int shared[];
    size_t *batch_buffer = (size_t*)(shared + 2);
    record_pair *out_buffer = (record_pair*)(batch_buffer + blockDim.x);

    int candidates = 0;
    const size_t stride = STRIDE();

    if (threadIdx.x < 2)
    {
        shared[threadIdx.x] = 0;
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
                    int old_count = atomicAdd(shared, 1);
                    if (old_count >= blockDim.x) { break; }
                    batch_buffer[old_count] = overlap_index;
                }
                pack = pack >> (sizeof(overlap_t) * CHAR_BIT);
                ++overlap_index;
            }

            __syncthreads();
            int batch = shared[0];
            __syncthreads();

            // read next packs if batch is not full and not the last one
            if (batch < blockDim.x && (block + stride < pack_count || batch == 0))
            {
                break;
            }

            // process current batch
            if (threadIdx.x == 0)
            {
                candidates += blockDim.x < batch ? blockDim.x : batch;
                shared[0] = 0;
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0) { atomicAdd(buffer_d, candidates); }
}
