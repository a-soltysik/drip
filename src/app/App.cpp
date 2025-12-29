#include "App.hpp"

#include <fmt/base.h>

#include <boost/exception/detail/exception_ptr.hpp>
#include <chrono>
#include <drip/common/log/LogMessageBuilder.hpp>
#include <drip/common/log/Logger.hpp>
#include <drip/common/log/sink/ConsoleSink.hpp>
#include <drip/common/log/sink/FileSink.hpp>
#include <glm/ext/vector_uint2.hpp>

#include "SystemSignalHandler.hpp"
#include "Window.hpp"

namespace drip::app
{

void App::run()
{
    registerSystemSignalHandlers();

    common::log::Logger::instance().addSink<common::log::ConsoleSink>();
    common::log::Logger::instance().addSink<common::log::FileSink>();
    common::log::Logger::instance().setAutoFlush(true, std::chrono::seconds {5});
    common::log::Logger::instance().setFlushOnLevel(common::log::Level::Error);
    common::log::Logger::instance().setExceptionHandler([](const boost::exception_ptr& ex) {
        fmt::println("{}",
                     common::log::Error("Logger exception!")
                         .withCurrentException(ex)
                         .withStacktraceFromCurrentException()
                         .asString());
    });
    common::log::Logger::instance().start();

    auto window = Window {
        glm::uvec2 {1280, 720},
        "drip::app"
    };

    while (!window.shouldClose())
    {
        window.processInput();
    }
}
}