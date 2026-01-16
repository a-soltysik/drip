#include <drip/common/log/LogMessageBuilder.hpp>

#include "App.hpp"

auto main(int argc, char** argv) -> int
{
    try
    {
        if (argc > 1)
        {
            drip::app::App {}.run(argv[1]);  //NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        }
        else
        {
            drip::app::App {}.run({});
        }
    }
    catch (...)
    {
        drip::common::log::Error("Unhandled critical exception")
            .withCurrentException()
            .withStacktraceFromCurrentException();
        return -1;
    }
}
