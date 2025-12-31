#pragma once

#include <GLFW/glfw3.h>

#include <glm/ext/vector_float2.hpp>

#include "ui/Window.hpp"

namespace drip::app
{

class RotationHandler
{
public:
    struct Controls
    {
        int rotationInitiation;
    };

    static constexpr auto controls = Controls {.rotationInitiation = GLFW_MOUSE_BUTTON_LEFT};

    explicit RotationHandler(const Window& window);
    [[nodiscard]] auto getRotation() const -> glm::vec2;

private:
    [[nodiscard]] auto getPixelsToAngleRatio() const -> glm::vec2;

    const Window& _window;
};

}
