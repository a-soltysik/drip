#pragma once

#include <fmt/base.h>

#include <drip/common/utils/format/GlmFormatter.hpp>
#include <string_view>

#include "drip/simulation/SimulationConfig.cuh"

namespace drip::sim
{

struct Fluid
{
    struct Properties
    {
        struct Particle
        {
            float mass;
            float radius;
            float smoothingRadius;
            float neighborSearchRangeSquared;
        };

        Particle particle;
        float density;
        float surfaceTension;
        float viscosity;
        float maxVelocity;
        float speedOfSound;
        uint32_t particleCount;
    };

    Bounds bounds;
    Properties properties;
};

struct SimulationParameters
{
    struct Domain
    {
        Bounds bounds;

        [[nodiscard]] auto getSize() const noexcept -> glm::vec3
        {
            return bounds.max - bounds.min;
        }
    };

    Domain domain;
    Fluid fluid;
    glm::vec3 gravity;
};

}

template <>
struct fmt::formatter<drip::sim::Bounds> : formatter<std::string_view>
{
    template <typename FormatContext>
    [[nodiscard]] auto format(const drip::sim::Bounds& bounds, FormatContext& ctx) const -> decltype(ctx.out())
    {
        return fmt::format_to(ctx.out(), "[min: {}, max: {}]", bounds.min, bounds.max);
    }
};

template <>
struct fmt::formatter<drip::sim::Fluid::Properties::Particle> : formatter<std::string_view>
{
    template <typename FormatContext>
    [[nodiscard]] auto format(const drip::sim::Fluid::Properties::Particle& p, FormatContext& ctx) const
        -> decltype(ctx.out())
    {
        return fmt::format_to(ctx.out(),
                              "Particle(mass: {}, radius: {}, smoothingRadius: {}, neighborSearchRangeSquared: {})",
                              p.mass,
                              p.radius,
                              p.smoothingRadius,
                              p.neighborSearchRangeSquared);
    }
};

template <>
struct fmt::formatter<drip::sim::Fluid::Properties> : formatter<std::string_view>
{
    template <typename FormatContext>
    [[nodiscard]] auto format(const drip::sim::Fluid::Properties& props, FormatContext& ctx) const
        -> decltype(ctx.out())
    {
        return fmt::format_to(ctx.out(),
                              "Properties(\n"
                              "    {}\n"
                              "    density: {}, surfaceTension: {}, viscosity: {}\n"
                              "    maxVelocity: {}, speedOfSound: {})",
                              props.particle,
                              props.density,
                              props.surfaceTension,
                              props.viscosity,
                              props.maxVelocity,
                              props.speedOfSound);
    }
};

template <>
struct fmt::formatter<drip::sim::Fluid> : formatter<std::string_view>
{
    template <typename FormatContext>
    [[nodiscard]] auto format(const drip::sim::Fluid& fluid, FormatContext& ctx) const -> decltype(ctx.out())
    {
        return fmt::format_to(ctx.out(),
                              "Fluid(\n"
                              "  bounds: {}\n"
                              "  {})",
                              fluid.bounds,
                              fluid.properties);
    }
};

template <>
struct fmt::formatter<drip::sim::SimulationParameters> : formatter<std::string_view>
{
    template <typename FormatContext>
    [[nodiscard]] auto format(const drip::sim::SimulationParameters& params, FormatContext& ctx) const
        -> decltype(ctx.out())
    {
        return fmt::format_to(ctx.out(),
                              "SimulationParameters(\n"
                              "  domain.bounds: {}\n"
                              "  {}\n"
                              "  gravity: {})",
                              params.domain.bounds,
                              params.fluid,
                              params.gravity);
    }
};
