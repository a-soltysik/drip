#include "App.hpp"

#include <fmt/base.h>

#include <boost/exception/detail/exception_ptr.hpp>
#include <chrono>
#include <drip/common/log/LogMessageBuilder.hpp>
#include <drip/common/log/Logger.hpp>
#include <drip/common/log/sink/ConsoleSink.hpp>
#include <drip/common/log/sink/FileSink.hpp>
#include <drip/engine/rendering/system/MeshRenderSystem.hpp>
#include <drip/engine/resource/MeshRenderable.hpp>
#include <drip/engine/resource/Surface.hpp>
#include <drip/engine/scene/Camera.hpp>
#include <drip/engine/scene/Light.hpp>
#include <drip/engine/utils/Signals.hpp>
#include <drip/engine/vulkan/core/Context.hpp>
#include <glm/ext/vector_uint2.hpp>
#include <glm/trigonometric.hpp>
#include <memory>
#include <utility>

#include "SystemSignalHandler.hpp"
#include "Window.hpp"
#include "mesh/InvertedCube.hpp"

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

    _window = std::make_unique<Window>(glm::uvec2 {1280, 720}, "drip::app");
    _api = std::make_unique<engine::gfx::Context>(*_window);
    _scene = std::make_unique<engine::gfx::Scene>();

    _api->addRenderSystem<engine::gfx::MeshRenderSystem>(_api->getDevice(), _api->getRenderer());
    initializeDefaultScene();
    mainLoop();
}

void App::initializeDefaultScene() const
{
    auto blueTexture = engine::gfx::Texture::getDefaultTexture(*_api, {0.25, 0.25, 0.3, 1.F});
    auto invertedCubeMesh = mesh::inverted_cube::create(*_api, "InvertedCube");
    auto invertedCubeRenderable = std::make_unique<engine::gfx::MeshRenderable>("InvertedCube");
    invertedCubeRenderable->addSurface(engine::gfx::Surface {blueTexture.get(), invertedCubeMesh.get()});
    invertedCubeRenderable->transform = engine::gfx::Transform {
        .translation = {},
        .scale = {1, 1, 1},
        .rotation = {},
    };

    _scene->addRenderable(std::move(invertedCubeRenderable));

    _api->registerMesh(std::move(invertedCubeMesh));
    _api->registerTexture(std::move(blueTexture));

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
    while (!_window->shouldClose()) [[likely]]
    {
        if (!_window->isMinimized()) [[likely]]
        {
            engine::signal::gameLoopIterationStarted.registerSender()();
            _window->processInput();

            _scene->getCamera().setPerspectiveProjection(
                engine::gfx::projection::Perspective {.fovY = glm::radians(50.F),
                                                      .aspect = _api->getAspectRatio(),
                                                      .zNear = 0.1F,
                                                      .zFar = 100});

            _api->makeFrame(*_scene);
        }
        else [[unlikely]]
        {
            _window->waitForInput();
        }
    }
}
}