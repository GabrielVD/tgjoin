#include <runtime_stats.h>

void ssjoin_stats::print(FILE *stream)
{
    if (status == ssjoin_status::SUCCESS)
    {
        fprintf(stream, "Host2Device\t%dms\n", host2device_ms);
    }
    else { fprintf(stream, "help\n"); }
}
