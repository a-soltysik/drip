#include <drip/common/log/LogMessageBuilder.hpp>

#include "App.hpp"

auto main(int /*argc*/, char** /*argv*/) -> int
{
    try
    {
        drip::app::App {}.run();
    }
    catch (...)
    {
        drip::common::log::Debug("Unhandled critical exception")
            .withCurrentException()
            .withStacktraceFromCurrentException();
        return -1;
    }
}