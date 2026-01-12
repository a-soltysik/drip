#pragma once
#include "Window.hpp"
#include "drip/gfx/resource/MeshRenderable.hpp"
#include "drip/gfx/scene/Camera.hpp"

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
                           gfx::Camera& camera,
                           const Config& config,
                           const gfx::Transform& initialTransform = {});
    void update(float deltaTime, float aspectRatio);

private:
    const Window& _window;
    gfx::Camera& _camera;
    gfx::Transform _transform;
    Config _config;
};

}
