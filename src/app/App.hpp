#pragma once
#include <drip/engine/Window.hpp>
#include <drip/engine/scene/Scene.hpp>
#include <drip/engine/vulkan/core/Context.hpp>

#include "drip/simulation/Simulation.cuh"
#include "ui/CameraHandler.hpp"

namespace drip::app
{
class App
{
public:
    void run();

private:
    void initializeDefaultScene();
    void mainLoop() const;

    std::unique_ptr<engine::Window> _window;
    std::unique_ptr<engine::gfx::Context> _api;
    std::unique_ptr<engine::gfx::Scene> _scene;
    std::unique_ptr<sim::Simulation> _simulation;
    std::unique_ptr<CameraHandler> _cameraHandler;
};
}
