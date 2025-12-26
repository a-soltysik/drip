#include <drip/common/Logger.hpp>
#include <glm/ext/vector_uint2.hpp>

#include "Window.hpp"

auto main(int /*argc*/, char** /*argv*/) -> int
{
    drip::common::log::Logger::instance().addSink<drip::common::log::ConsoleSink>();
    drip::common::log::Logger::instance().addSink<drip::common::log::FileSink>();
    drip::common::log::Logger::instance().start();

    auto window = drip::app::Window {
        glm::uvec2 {1280, 720},
        "drip::app"
    };

    while (!window.shouldClose())
    {
        window.processInput();
    }
}