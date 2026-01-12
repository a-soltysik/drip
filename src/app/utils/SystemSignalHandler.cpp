#include "SystemSignalHandler.hpp"

#include <csignal>
#include <cstdlib>
#include <drip/common/log/LogMessageBuilder.hpp>
#include <drip/common/log/Logger.hpp>
#include <drip/common/utils/Assert.hpp>
#include <string_view>

namespace drip::app::utils
{

namespace
{
[[nodiscard]] constexpr auto getSignalName(const int signalValue) noexcept -> std::string_view
{
    switch (signalValue)
    {
    case SIGABRT:
        return "SIGABRT";
    case SIGFPE:
        return "SIGFPE";
    case SIGILL:
        return "SIGILL";
    case SIGINT:
        return "SIGINT";
    case SIGSEGV:
        return "SIGSEGV";
    case SIGTERM:
        return "SIGTERM";
    default:
        return "unknown";
    }
}

[[noreturn]] void signalHandler(const int signalValue)
{
    common::log::Error("Received {} signal", getSignalName(signalValue)).withStacktrace();
    common::log::Logger::instance().stop();
    std::_Exit(signalValue);
}
}

void registerSystemSignalHandlers()
{
    common::ShouldNotBe(std::signal(SIGABRT, signalHandler), SIG_ERR, "Failed to register signal handler");
    common::ShouldNotBe(std::signal(SIGFPE, signalHandler), SIG_ERR, "Failed to register signal handler");
    common::ShouldNotBe(std::signal(SIGILL, signalHandler), SIG_ERR, "Failed to register signal handler");
    common::ShouldNotBe(std::signal(SIGINT, signalHandler), SIG_ERR, "Failed to register signal handler");
    common::ShouldNotBe(std::signal(SIGSEGV, signalHandler), SIG_ERR, "Failed to register signal handler");
    common::ShouldNotBe(std::signal(SIGTERM, signalHandler), SIG_ERR, "Failed to register signal handler");
}

}
