#include <helper_mem.cuh>
#include <helper_interface.h>
#include <ssjoin_staging.h>
#include <ssjoin.h>

int main(int argc, char **argv)
{
    input_info info;
    info.threshold = FLOAT_ARG("j");
    if (argc < 2 || info.threshold == .0f || HAS_ARG("help") || HAS_ARG("h"))
    {
        print_help();
        exit(EXIT_WAIVED);
    }

    info.mem_min = INT_ARG("mem");
    info.mem_min *= 1000000;
    if (info.mem_min == 0) { info.mem_min = 100000000; }

    fprintf(stderr,
            "Threshold" TABS "Jaccard %.2f\n"
            "Minimum free GPU memory\t%ldMB\n",
            info.threshold, info.mem_min / 1000000);
    for (int i = 1; i < argc; ++i)
    {
        if (is_option((const char *)argv[i])) { continue; }
        
        fprintf(stderr, "Document: %s\n", argv[i]);
        info.pathname = argv[i];
        run_join(info).print(stderr);
        break;
    }
    
    return EXIT_SUCCESS;
}
