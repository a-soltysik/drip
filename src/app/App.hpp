#pragma once
#include <drip/gfx/vulkan/core/Context.hpp>
#include <drip/simulation/Simulation.cuh>

#include "ui/CameraHandler.hpp"
#include "utils/Scene.hpp"

namespace drip::app
{
class App
{
public:
    void run();

private:
    static void initializeLogger();
    void mainLoop() const;

    std::unique_ptr<Window> _window;
    std::unique_ptr<gfx::Context> _api;
    std::unique_ptr<utils::Scene> _scene;
    std::unique_ptr<sim::Simulation> _simulation;
    std::unique_ptr<CameraHandler> _cameraHandler;
};
}
