#pragma once

#include <glm/ext/vector_float3.hpp>
#include <glm/vec3.hpp>

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
        Bounds bounds;
    };

    struct Fluid
    {
        struct Properties
        {
            float spacing;
            float smoothingRadius;
            float density;
            float surfaceTension;
            float viscosity;
            float maxVelocity;
            float speedOfSound;
        };

        Bounds bounds;
        Properties properties;
    };

    Domain domain;
    Fluid fluid;
    glm::vec3 gravity;
};

inline const auto defaultSimulationParameters = SimulationConfig {
    .domain = {.bounds = {.min = {-1.F, -1.F, -1.F}, .max = {1.F, 1.F, 1.F}}},
    .fluid = {.bounds = {.min = {-1.F, -1.F, -1.F}, .max = {1.F, 1.F, 1.F}},
               .properties = {.spacing = 0.08F,
                             .smoothingRadius = 0.08F,
                             .density = 1000.0F,
                             .surfaceTension = 1.0F,
                             .viscosity = 0.1F,
                             .maxVelocity = 10.0F,
                             .speedOfSound = 50.0F}},
    .gravity = {0.F, -9.81F, 0.F}
};

}
