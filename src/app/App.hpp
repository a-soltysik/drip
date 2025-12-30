#pragma once
#include <drip/engine/Window.hpp>
#include <drip/engine/scene/Scene.hpp>
#include <drip/engine/vulkan/core/Context.hpp>

namespace drip::app
{
class App
{
public:
    void run();

private:
    void initializeDefaultScene() const;
    void mainLoop() const;

    std::unique_ptr<engine::gfx::Scene> _scene;
    std::unique_ptr<engine::Window> _window;
    std::unique_ptr<engine::gfx::Context> _api;
};
}
