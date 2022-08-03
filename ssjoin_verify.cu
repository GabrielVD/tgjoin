#include <ssjoin_verify.cuh>
#include <helper_cuda.h>

#define FILL ((overlap_t)-1)

__global__ void verify(
    int* __restrict__ buffer_d,
    overlap_pack* __restrict__ overlap_pack_d,
    const int pack_count)
{
    extern __shared__ int shared[];

    if (threadIdx.x == 0) { shared[0] = 0; }

    const int stride = STRIDE();
    int candidates = 0;
    for (int idx = IDX(); idx < pack_count; idx += stride)
    {
        __syncthreads();
        auto pack = atomicExch(overlap_pack_d + idx, 0);

        while (1)
        {
            while (pack != 0)
            {
                overlap_t overlap = pack & FILL;
                if (overlap != 0 && overlap != FILL)
                {
                    if (atomicAdd(shared, 1) >= blockDim.x) { break; }
                }
                pack = pack >> (sizeof(overlap_t) * CHAR_BIT);
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
