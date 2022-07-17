#include <helper_mem.cuh>
#include <helper_interface.h>
#include <ssjoin_staging.h>
#include <ssjoin.h>

int main(int argc, char **argv)
{
    const auto threshold{FLOAT_ARG("j")};
    if (argc < 2 || threshold == .0f || HAS_ARG("help") || HAS_ARG("h"))
    {
        print_help();
        exit(EXIT_WAIVED);
    }

    size_t mem_min = INT_ARG("mem") * 1000000;
    if (mem_min == 0) { mem_min = 100000000; }

    fprintf(stderr,
            "Threshold" TABS "Jaccard %.2f\n"
            "Minimum free GPU memory\t%ldMB\n",
            threshold, mem_min / 1000000);
    for (int i = 1; i < argc; ++i)
    {
        if (is_option((const char *)argv[i])) { continue; }
        
        record_t *dataset;
        input_info info;
        info.threshold = threshold;
        info.mem_min = mem_min;
        if (load_dataset(argv[i], &dataset, &info.data_size) == 0)
        {
            if (verify_dataset(dataset, info) == 0) {
                info.print(stderr);
                run_join(dataset, info).print(stderr);
            }
            else { print_formaterr(); }
            SAFE_FREE(&dataset, &info.data_size);
        }
    }
    
    return EXIT_SUCCESS;
}
