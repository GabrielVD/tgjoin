#include <ssjoin_shared.h>
#include <helper_cuda.h>
#include <helper_mem.cuh>

void transfer_records_async(
    uint32_t **records_out,
    uint32_t **records_d_out,
    size_t &buffer_size,
    const uint32_t *input,
    size_t input_size,
    int cardinality)
{
    buffer_size = input_size - cardinality + 1;
    const size_t bytes{BYTES_U(buffer_size)};
    checkCudaErrors(cudaMalloc(records_d_out, bytes));
    checkCudaErrors(cudaMallocHost(records_out, bytes));

    uint32_t *records{*records_out}, *records_d{*records_d_out};
    {
        size_t set_start = cardinality + 1;
        records[0] = set_start;
        for (size_t i{1}, i_input{1}; i_input < input_size; ++i)
        {
            size_t set_size{input[i_input]};
            memcpy(records + set_start, input + i_input + 1, BYTES_U(set_size));

            i_input += set_size + 2;
            set_start += set_size;
            records[i] = set_start;
        }
    }

    checkCudaErrors(cudaMemcpyAsync(records_d, records, bytes, cudaMemcpyHostToDevice));
}
