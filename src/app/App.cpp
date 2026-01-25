#include "App.hpp"

#include <fmt/base.h>

#include <array>
#include <boost/exception/detail/exception_ptr.hpp>
#include <chrono>
#include <drip/common/log/LogMessageBuilder.hpp>
#include <drip/common/log/Logger.hpp>
#include <drip/common/log/sink/ConsoleSink.hpp>
#include <drip/common/log/sink/FileSink.hpp>
#include <drip/common/utils/format/GlmFormatter.hpp>  //NOLINT(misc-include-cleaner)
#include <drip/gfx/resource/MeshRenderable.hpp>
#include <drip/gfx/resource/ParticlesRenderable.hpp>
#include <drip/gfx/vulkan/core/Context.hpp>
#include <drip/simulation/ExternalMemory.cuh>
#include <drip/simulation/Simulation.cuh>
#include <drip/simulation/SimulationConfig.cuh>
#include <filesystem>
#include <glm/ext/vector_float4.hpp>
#include <glm/ext/vector_uint2.hpp>
#include <memory>
#include <optional>

#include "config/JsonFileReader.hpp"
#include "ui/CameraHandler.hpp"
#include "ui/Window.hpp"
#include "ui/panel/StatisticsPanel.hpp"
#include "utils/FrameTimeManager.hpp"
#include "utils/Scene.hpp"
#include "utils/Signals.hpp"
#include "utils/SystemSignalHandler.hpp"

namespace drip::app
{

App::App(std::optional<std::filesystem::path> configurationFile)
{
    utils::registerSystemSignalHandlers();
    initializeLogger();

    const auto simulationParameters = configurationFile
                                          .and_then([](const auto& filePath) {
                                              return readJsonFile<sim::SimulationConfig>(filePath);
                                          })
                                          .value_or(sim::defaultSimulationParameters);

    _window = std::make_unique<Window>(glm::uvec2 {1280, 720}, "drip::app");
    _gfxContext = std::make_unique<gfx::Context>(*_window);
    _gfxContext->getGuiManager().addPanel<StatisticsPanel>();
    _scene = utils::Scene::createDefaultScene(*_gfxContext, simulationParameters);
    _cameraHandler = std::make_unique<CameraHandler>(*_window,
                                                     _scene->getGfxScene().getCamera(),
                                                     CameraHandler::Config {
                                                         .rotationSpeed = 500.F,
                                                         .moveSpeed = 2.5F
    },
                                                     gfx::Transform {.translation = {0, 0.5F, -5}});

    const auto& fluidParticles = _scene->getFluidParticles();
    const auto sharedBuffer = fluidParticles.getDataBuffer();
    const auto particleCount = fluidParticles.getSize();
    _simulation = sim::Simulation::create(
        sim::Simulation::SharedMemory {
            .positions = sim::ExternalMemory::create(sharedBuffer.translations, particleCount * sizeof(glm::vec4)),
            .colors = sim::ExternalMemory::create(sharedBuffer.colors, particleCount * sizeof(glm::vec4)),
            .sizes = sim::ExternalMemory::create(sharedBuffer.sizes, particleCount * sizeof(float))},
        simulationParameters);
}

void App::run() const
{
    mainLoop();
}

void App::initializeLogger()
{
    common::log::Logger::instance().addSink<common::log::ConsoleSink>();
    common::log::Logger::instance().addSink<common::log::FileSink>();
    common::log::Logger::instance().setAutoFlush(true, std::chrono::seconds {5});
    common::log::Logger::instance().setLevels(std::array {common::log::Logger::Level::Info,
                                                          common::log::Logger::Level::Warning,
                                                          common::log::Logger::Level::Error});
    common::log::Logger::instance().setFlushOnLevel(common::log::Logger::Level::Error);
    common::log::Logger::instance().setExceptionHandler([](const boost::exception_ptr& ex) {
        fmt::println("{}",
                     common::log::Error("Logger exception!")
                         .withCurrentException(ex)
                         .withStacktraceFromCurrentException()
                         .asString());
    });
    common::log::Logger::instance().start();
}

void App::mainLoop() const
{
    auto timeManager = utils::FrameTimeManager {};

    while (!_window->shouldClose()) [[likely]]
    {
        if (!_window->isMinimized()) [[likely]]
        {
            signal::mainLoopIterationStarted.registerSender()();
            _window->processInput();

            timeManager.update();

            _cameraHandler->update(timeManager.getDelta(), _gfxContext->getAspectRatio());

            _simulation->update(_simulation->getCflTimestep());

            _gfxContext->makeFrame(_scene->getGfxScene());
        }
        else [[unlikely]]
        {
            _window->waitForInput();
        }
    }

    common::log::Info("Mean frame rate: {}", timeManager.getMeanFrameRate());
}
}
