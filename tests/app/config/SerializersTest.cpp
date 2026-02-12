#include <doctest/doctest.h>

#include <drip/simulation/SimulationConfig.cuh>
#include <glm/ext/vector_float3.hpp>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

#include "config/Serializers.hpp"  //NOLINT(misc-include-cleaner)

TEST_SUITE("Serializers")
{
    TEST_CASE("glm::vec3 serialization")
    {
        const auto vec = glm::vec3 {1.0F, 2.0F, 3.0F};

        auto j = nlohmann::json(vec);

        CHECK(j["x"] == 1.0F);
        CHECK(j["y"] == 2.0F);
        CHECK(j["z"] == 3.0F);
    }

    TEST_CASE("glm::vec3 deserialization")
    {
        const auto j = nlohmann::json {
            {"x", 4.0F},
            {"y", 5.0F},
            {"z", 6.0F}
        };

        const auto vec = j.get<glm::vec3>();

        CHECK(vec.x == 4.0F);
        CHECK(vec.y == 5.0F);
        CHECK(vec.z == 6.0F);
    }

    TEST_CASE("glm::vec3 round-trip")
    {
        static constexpr auto original = glm::vec3 {-1.5F, 0.0F, 42.0F};

        const auto j = nlohmann::json(original);
        const auto restored = j.get<glm::vec3>();

        CHECK(restored.x == original.x);
        CHECK(restored.y == original.y);
        CHECK(restored.z == original.z);
    }

    TEST_CASE("Bounds serialization")
    {
        static constexpr auto bounds = drip::sim::Bounds {
            .min = {-1.0F, -2.0F, -3.0F},
            .max = {1.0F,  2.0F,  3.0F }
        };

        const auto j = nlohmann::json(bounds);

        CHECK(j["min"]["x"] == -1.0F);
        CHECK(j["min"]["y"] == -2.0F);
        CHECK(j["min"]["z"] == -3.0F);
        CHECK(j["max"]["x"] == 1.0F);
        CHECK(j["max"]["y"] == 2.0F);
        CHECK(j["max"]["z"] == 3.0F);
    }

    TEST_CASE("Bounds deserialization")
    {
        const auto j = nlohmann::json {
            {"min", {{"x", 0.0F}, {"y", 0.0F}, {"z", 0.0F}}},
            {"max", {{"x", 5.0F}, {"y", 5.0F}, {"z", 5.0F}}}
        };

        const auto bounds = j.get<drip::sim::Bounds>();

        CHECK(bounds.min.x == 0.0F);
        CHECK(bounds.max.x == 5.0F);
    }

    TEST_CASE("SimulationConfig serialization round-trip")
    {
        auto config = drip::sim::SimulationConfig {};
        config.domain.bounds.min = {-2.0F, -2.0F, -2.0F};
        config.domain.bounds.max = {2.0F, 2.0F, 2.0F};
        config.fluid.properties.spacing = 0.1F;
        config.fluid.properties.density = 500.0F;
        config.environment.gravity = {0.0F, -5.0F, 0.0F};

        const auto j = nlohmann::json(config);
        const auto restored = j.get<drip::sim::SimulationConfig>();

        CHECK(restored.domain.bounds.min.x == -2.0F);
        CHECK(restored.domain.bounds.max.x == 2.0F);
        CHECK(restored.fluid.properties.spacing == 0.1F);
        CHECK(restored.fluid.properties.density == 500.0F);
        CHECK(restored.environment.gravity.y == -5.0F);
    }

    TEST_CASE("SimulationConfig uses defaults for missing fields")
    {
        const auto j = nlohmann::json {
            {"domain",
             {{"bounds", {{"min", {{"x", 0}, {"y", 0}, {"z", 0}}}, {"max", {{"x", 1}, {"y", 1}, {"z", 1}}}}}}               },
            {"fluid",       {{"bounds", {{"min", {{"x", 0}, {"y", 0}, {"z", 0}}}, {"max", {{"x", 1}, {"y", 1}, {"z", 1}}}}}}},
            {"environment", {}                                                                                              }
        };

        const auto config = j.get<drip::sim::SimulationConfig>();

        CHECK(config.fluid.properties.spacing == 0.08F);
        CHECK(config.fluid.properties.density == 1000.0F);
        CHECK(config.environment.gravity.y == -9.81F);
    }
}