#pragma once

#include <glm/ext/vector_float3.hpp>

namespace drip::sim
{

struct Bounds
{
    glm::vec3 min;
    glm::vec3 max;
};

struct SimulationConfig
{
    struct Domain
    {
        Bounds bounds = {
            .min = {-1.F, -1.F, -1.F},
              .max = {1.F,  1.F,  1.F }
        };
    };

    struct Fluid
    {
        struct Properties
        {
            float spacing = 0.08F;
            float smoothingRadius = 0.08F;
            float density = 1000.0F;
            float surfaceTension = 1.0F;
            float viscosity = 0.1F;
            float maxVelocity = 10.0F;
            float speedOfSound = 50.0F;
        };

        Bounds bounds = {
            .min = {-1.F, -1.F, -1.F},
              .max = {1.F,  1.F,  1.F }
        };
        Properties properties;
    };

    struct Environment
    {
        glm::vec3 gravity = {0.F, -9.81F, 0.F};
    };

    Domain domain {};
    Fluid fluid {};
    Environment environment {};
};

}
