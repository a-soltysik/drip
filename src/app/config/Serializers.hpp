#pragma once

#include <cstddef>
#include <glm/detail/qualifier.hpp>
#include <glm/vec3.hpp>
#include <nlohmann/adl_serializer.hpp>
#include <nlohmann/detail/macro_scope.hpp>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

namespace nlohmann
{

template <glm::length_t L, typename T, glm::qualifier Q>
struct adl_serializer<glm::vec<L, T, Q>>
{
    static void to_json(json& j, const glm::vec<L, T, Q>& vec)
    {
        j = json::array();
        for (auto i = glm::length_t {0}; i < L; ++i)
        {
            j.push_back(vec[i]);
        }
    }

    static void from_json(const json& j, glm::vec<L, T, Q>& vec)
    {
        for (auto i = glm::length_t {0}; i < L; ++i)
        {
            vec[i] = j[static_cast<size_t>(i)].get<T>();
        }
    }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(drip::sim::Bounds, min, max)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(drip::sim::SimulationConfig::Domain, bounds)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(drip::sim::SimulationConfig::Fluid::Properties,
                                   spacing,
                                   smoothingRadius,
                                   density,
                                   surfaceTension,
                                   viscosity,
                                   maxVelocity,
                                   speedOfSound)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(drip::sim::SimulationConfig::Fluid, bounds, properties)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(drip::sim::SimulationConfig, domain, fluid, gravity)
}
