#pragma once
#include "Window.hpp"
#include "drip/engine/resource/MeshRenderable.hpp"
#include "drip/engine/scene/Camera.hpp"

namespace drip::app
{

class CameraHandler
{
public:
    struct Config
    {
        float rotationSpeed;
        float moveSpeed;
    };

    explicit CameraHandler(const Window& window,
                           engine::gfx::Camera& camera,
                           const Config& config,
                           const engine::gfx::Transform& initialTransform = {});
    void update(float deltaTime, float aspectRatio);

private:
    const Window& _window;
    engine::gfx::Camera& _camera;
    engine::gfx::Transform _transform;
    Config _config;
};

}
