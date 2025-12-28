#include "App.hpp"

#include <glm/ext/vector_uint2.hpp>

#include "SystemSignalHandler.hpp"
#include "Window.hpp"
#include "drip/common/Logger.hpp"

namespace drip::app
{

void App::run()
{
    registerSystemSignalHandlers();

    common::log::Logger::instance().addSink<common::log::ConsoleSink>();
    common::log::Logger::instance().addSink<common::log::FileSink>();
    common::log::Logger::instance().start();

    auto window = app::Window {
        glm::uvec2 {1280, 720},
        "drip::app"
    };

    while (!window.shouldClose())
    {
        window.processInput();
    }
}
}