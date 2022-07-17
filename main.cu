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

    for (int i = 1; i < argc; ++i)
    {
        if (is_option((const char *)argv[i])) { continue; }
        
        uint32_t *dataset;
        input_info info;
        info.threshold = threshold;
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
