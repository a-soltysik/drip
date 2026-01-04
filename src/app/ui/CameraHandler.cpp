#include "CameraHandler.hpp"

#include <drip/engine/resource/MeshRenderable.hpp>
#include <drip/engine/scene/Camera.hpp>
#include <glm/common.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/trigonometric.hpp>

#include "MovementHandler.hpp"
#include "RotationHandler.hpp"
#include "ui/Window.hpp"

namespace drip::app
{

CameraHandler::CameraHandler(const Window& window,
                             engine::gfx::Camera& camera,
                             const Config& config,
                             const engine::gfx::Transform& initialTransform)
    : _window(window),
      _camera(camera),
      _transform(initialTransform),
      _config(config)
{
}

void CameraHandler::update(float deltaTime, float aspectRatio)
{
    _camera.setPerspectiveProjection(engine::gfx::projection::Perspective {.fovY = glm::radians(50.F),
                                                                           .aspect = aspectRatio,
                                                                           .zNear = 0.1F,
                                                                           .zFar = 100});
    _transform.rotation += glm::vec3 {RotationHandler {_window}.getRotation() * _config.rotationSpeed * deltaTime, 0};

    _transform.rotation.x = glm::clamp(_transform.rotation.x, -glm::half_pi<float>(), glm::half_pi<float>());
    _transform.rotation.y = glm::mod(_transform.rotation.y, glm::two_pi<float>());

    const auto rawMovement = MovementHandler {_window}.getMovement();

    const auto cameraDirection = glm::vec3 {glm::cos(-_transform.rotation.x) * glm::sin(_transform.rotation.y),
                                            glm::sin(-_transform.rotation.x),
                                            glm::cos(-_transform.rotation.x) * glm::cos(_transform.rotation.y)};
    const auto cameraRight = glm::vec3 {glm::cos(_transform.rotation.y), 0, -glm::sin(_transform.rotation.y)};

    auto translation = cameraDirection * rawMovement.z.value_or(0);
    translation += cameraRight * rawMovement.x.value_or(0);
    translation.y = rawMovement.y.value_or(translation.y);

    if (glm::dot(translation, translation) > 0)
    {
        _transform.translation += glm::normalize(translation) * _config.moveSpeed * deltaTime;
    }

    _camera.setViewYXZ(engine::gfx::view::YXZ {
        .position = _transform.translation,
        .rotation = {-_transform.rotation.x, _transform.rotation.y, 0}
    });
}

}
