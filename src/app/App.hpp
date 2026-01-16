#pragma once
#include <drip/gfx/vulkan/core/Context.hpp>
#include <drip/simulation/Simulation.cuh>
#include <filesystem>
#include <memory>
#include <optional>

#include "ui/CameraHandler.hpp"
#include "ui/Window.hpp"
#include "utils/Scene.hpp"

namespace drip::app
{
class App
{
public:
    void run(std::optional<std::filesystem::path> configurationFile);

private:
    static void initializeLogger();
    void mainLoop() const;

    std::unique_ptr<Window> _window;
    std::unique_ptr<gfx::Context> _gfxContext;
    std::unique_ptr<utils::Scene> _scene;
    std::unique_ptr<sim::Simulation> _simulation;
    std::unique_ptr<CameraHandler> _cameraHandler;
};
}
