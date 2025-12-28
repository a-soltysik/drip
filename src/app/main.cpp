#include <drip/common/Logger.hpp>

#include "App.hpp"

auto main(int /*argc*/, char** /*argv*/) -> int
{
    try
    {
        drip::app::App::run();
    }
    catch (...)
    {
        drip::common::log::Exception("Unhandled critical exception");  //NOLINT(bugprone-throw-keyword-missing)
        return -1;
    }
}