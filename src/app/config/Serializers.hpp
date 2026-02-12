#pragma once

#include <cstddef>
#include <drip/simulation/SimulationConfig.cuh>
#include <glm/detail/qualifier.hpp>
#include <glm/vec3.hpp>
#include <nlohmann/adl_serializer.hpp>
#include <nlohmann/detail/macro_scope.hpp>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

namespace nlohmann
{

template <typename T, glm::qualifier Q>
struct adl_serializer<glm::vec<3, T, Q>>
{
    static void to_json(json& j, const glm::vec<3, T, Q>& vec)
    {
        j = json {
            {"x", vec.x},
            {"y", vec.y},
            {"z", vec.z}
        };
    }

    static void from_json(const json& j, glm::vec<3, T, Q>& vec)
    {
        j.at("x").get_to(vec.x);
        j.at("y").get_to(vec.y);
        j.at("z").get_to(vec.z);
    }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(drip::sim::Bounds, min, max)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(drip::sim::SimulationConfig::Domain, bounds)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(drip::sim::SimulationConfig::Fluid::Properties,
                                                spacing,
                                                smoothingRadius,
                                                density,
                                                surfaceTension,
                                                viscosity,
                                                maxVelocity,
                                                speedOfSound)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(drip::sim::SimulationConfig::Environment, gravity)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(drip::sim::SimulationConfig::Fluid, bounds, properties)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(drip::sim::SimulationConfig, domain, fluid, environment)
}
