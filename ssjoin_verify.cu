#include <ssjoin_verify.cuh>
#include <helper_cuda.h>

__global__ void verify(
    overlap_t *overlap_matrix_d,
    const size_t overlap_count)
{
    extern __shared__ unsigned int histo_s[];
    const size_t stride = STRIDE();

    auto *packed_d = (unsigned long long*)overlap_matrix_d;
    for (size_t idx = IDX(); idx < overlap_count / VERIFY_PACK_SIZE; idx += stride)
    {
        __syncthreads();
        // volatile overlap_t overlap = atomicExch(packed_d + idx, 0);
        volatile overlap_t overlap = packed_d[idx];
        __syncthreads();
        packed_d[idx] = overlap;
    }
}
