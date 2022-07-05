#include <runtime_stats.h>

void ssjoin_stats::print(FILE *stream)
{
    if (status == ssjoin_status::SUCCESS)
    {
        fprintf(stream,
                "Host to device\t%dms\n"
                "Indexing\t%dms\n",
                host2device_ms,
                indexing_ms);
    }
    else
    {
        fprintf(stream, "Error\n");
    }
}
