#include <runtime_stats.h>

void ssjoin_stats::print(FILE *stream)
{
    if (status == ssjoin_status::SUCCESS)
    {
        fprintf(stream,
                "Host to device" TABS "%dms\n"
                "Indexing" TABS "%dms\n"
                "Max indexed token\t%d\n",
                host2device_ms,
                indexing_ms,
                max_indexed_token);
    }
    else
    {
        fprintf(stream, "Error\n");
    }
}
