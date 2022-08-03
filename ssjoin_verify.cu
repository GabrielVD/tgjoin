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
    size_t *buffer = ((size_t*)shared) + 1;

    if (threadIdx.x == 0) { shared[0] = 0; }

    const size_t stride = STRIDE();
    int candidates = 0;
    for (size_t idx = IDX(); idx < pack_count; idx += stride)
    {
        __syncthreads();
        size_t overlap_index = idx * OVERLAP_PACK_SIZE;
        auto pack = atomicExch(overlap_pack_d + idx, 0);

        while (1)
        {
            while (pack != 0)
            {
                overlap_t overlap = pack & FILL;
                if (overlap != 0 && overlap != FILL)
                {
                    int old_count = atomicAdd(shared, 1);
                    if (old_count >= blockDim.x) { break; }
                    buffer[old_count] = overlap_index;
                }
                pack = pack >> (sizeof(overlap_t) * CHAR_BIT);
                ++overlap_index;
            }

            __syncthreads();
            if (shared[0] < blockDim.x) { break; }
            __syncthreads();
            if (threadIdx.x == 0)
            {
                candidates += blockDim.x;
                shared[0] = 0;
            }
        }
    }

    if (threadIdx.x == 0) { atomicAdd(buffer_d, shared[0] + candidates); }
}
