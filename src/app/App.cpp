#include "App.hpp"

#include <fmt/base.h>

#include <array>
#include <boost/exception/detail/exception_ptr.hpp>
#include <chrono>
#include <drip/common/log/LogMessageBuilder.hpp>
#include <drip/common/log/Logger.hpp>
#include <drip/common/log/sink/ConsoleSink.hpp>
#include <drip/common/log/sink/FileSink.hpp>
#include <drip/engine/resource/MeshRenderable.hpp>
#include <drip/engine/resource/Surface.hpp>
#include <drip/engine/scene/Camera.hpp>
#include <drip/engine/scene/Light.hpp>
#include <drip/engine/vulkan/core/Context.hpp>
#include <drip/simulation/Simulation.cuh>
#include <glm/ext/vector_float4.hpp>
#include <glm/ext/vector_uint2.hpp>
#include <memory>
#include <utility>

#include "SystemSignalHandler.hpp"
#include "drip/engine/resource/ParticlesRenderable.hpp"
#include "drip/simulation/ExternalMemory.cuh"
#include "mesh/InvertedCube.hpp"
#include "ui/CameraHandler.hpp"
#include "ui/Window.hpp"
#include "ui/panel/StatisticsPanel.hpp"
#include "utils/FrameTimeManager.hpp"
#include "utils/Signals.hpp"

namespace drip::app
{

void App::run()
{
    registerSystemSignalHandlers();

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

    _window = std::make_unique<Window>(glm::uvec2 {1280, 720}, "drip::app");
    _api = std::make_unique<engine::gfx::Context>(*_window);
    _api->getGuiManager().addPanel<StatisticsPanel>();
    _scene = std::make_unique<engine::gfx::Scene>();
    _cameraHandler = std::make_unique<CameraHandler>(dynamic_cast<Window&>(*_window),
                                                     _scene->getCamera(),
                                                     CameraHandler::Config {
                                                         .rotationSpeed = 500.F,
                                                         .moveSpeed = 2.5F
    },
                                                     engine::gfx::Transform {.translation = {0, 0.5F, -5}});

    initializeDefaultScene();
    mainLoop();
}

void App::initializeDefaultScene()
{
    auto blueTexture = engine::gfx::Texture::getDefaultTexture(*_api, {0.25, 0.25, 0.3, 1.F});
    auto invertedCubeMesh = mesh::inverted_cube::create(*_api, "InvertedCube");
    auto invertedCubeRenderable = std::make_unique<engine::gfx::MeshRenderable>("InvertedCube");
    invertedCubeRenderable->addSurface(engine::gfx::Surface {blueTexture.get(), invertedCubeMesh.get()});

    _scene->addRenderable(std::move(invertedCubeRenderable));

    _api->registerMesh(std::move(invertedCubeMesh));
    _api->registerTexture(std::move(blueTexture));

    static constexpr auto particleCount = 1000ZU;
    auto fluidParticlesRenderable =
        std::make_unique<engine::gfx::ParticlesRenderable>(*_api, "FluidParticles", particleCount);
    const auto sharedMemory = fluidParticlesRenderable->getDataBuffer();

    common::log::Info("Initializing simulation with {} particles", particleCount);
    _simulation = sim::Simulation::create(sim::Simulation::SharedMemory {
        .positions = sim::ExternalMemory::create(sharedMemory.translations, particleCount * sizeof(glm::vec4)),
        .colors = sim::ExternalMemory::create(sharedMemory.colors, particleCount * sizeof(glm::vec4)),
        .sizes = sim::ExternalMemory::create(sharedMemory.sizes, particleCount * sizeof(float))});

    common::log::Info("Initialized simulation with {} particles", particleCount);

    _scene->addRenderable(std::move(fluidParticlesRenderable));

    auto directionalLight = engine::gfx::DirectionalLight {};
    directionalLight.name = "DirectionalLight";
    directionalLight.direction = {-6.2F, -2.F, -1.F};
    directionalLight.makeColorLight({1.F, .8F, .8F}, 0.F, 0.8F, 1.F, 0.8F);

    _scene->addLight(std::move(directionalLight));

    auto cameraObject = engine::gfx::Transform {
        .translation = {0, 0.5, -5}
    };
    _scene->getCamera().setViewYXZ(
        engine::gfx::view::YXZ {.position = cameraObject.translation, .rotation = cameraObject.rotation});
}

void App::mainLoop() const
{
    auto timeManager = FrameTimeManager {};

    while (!_window->shouldClose()) [[likely]]
    {
        if (!_window->isMinimized()) [[likely]]
        {
            signal::mainLoopIterationStarted.registerSender()();
            _window->processInput();

            timeManager.update();

            _cameraHandler->update(timeManager.getDelta(), _api->getAspectRatio());

            _simulation->update(timeManager.getDelta());

            _api->makeFrame(*_scene);
        }
        else [[unlikely]]
        {
            _window->waitForInput();
        }
    }

    common::log::Info("Mean frame rate: {}", timeManager.getMeanFrameRate());
}
}
