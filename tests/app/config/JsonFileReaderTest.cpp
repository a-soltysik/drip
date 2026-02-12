#include <doctest/doctest.h>

#include <drip/simulation/SimulationConfig.cuh>
#include <filesystem>
#include <fstream>
#include <nlohmann/json_fwd.hpp>
#include <string_view>

#include "config/JsonFileReader.hpp"

namespace
{

class TempFile
{
public:
    explicit TempFile(std::string_view filename)
        : _path(std::filesystem::temp_directory_path() / filename)
    {
    }

    TempFile(const TempFile&) = delete;
    auto operator=(const TempFile&) = delete;
    TempFile(TempFile&&) = delete;
    auto operator=(TempFile&&) = delete;

    ~TempFile()
    {
        std::filesystem::remove(_path);
    }

    [[nodiscard]] auto path() const -> const std::filesystem::path&
    {
        return _path;
    }

    void write(std::string_view content) const
    {
        std::ofstream file(_path);
        file << content;
    }

private:
    std::filesystem::path _path;
};

const auto minimalValidConfig = R"({
    "domain": {},
    "fluid": {},
    "environment": {}
})";

const auto fullConfig = R"({
    "domain": {
        "bounds": {
            "min": { "x": -2.0, "y": -2.0, "z": -2.0 },
            "max": { "x": 2.0, "y": 2.0, "z": 2.0 }
        }
    },
    "fluid": {
        "bounds": {
            "min": { "x": -1.0, "y": -1.0, "z": -1.0 },
            "max": { "x": 1.0, "y": 1.0, "z": 1.0 }
        },
        "properties": {
            "spacing": 0.05,
            "smoothingRadius": 0.1,
            "density": 998.0,
            "surfaceTension": 0.5,
            "viscosity": 0.2,
            "maxVelocity": 15.0,
            "speedOfSound": 60.0
        }
    },
    "environment": {
        "gravity": { "x": 0.0, "y": -10.0, "z": 0.0 }
    }
})";

const auto minimalValidSchema = R"({
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["domain", "fluid", "environment"]
})";

}

TEST_SUITE("JsonFileReader")
{
    TEST_CASE("readJsonFile returns nullopt for non-existent file")
    {
        const auto schema = TempFile("test_schema.json");
        schema.write(minimalValidSchema);

        const auto result = drip::app::readJsonFile<drip::sim::SimulationConfig>(
            std::filesystem::temp_directory_path() / "non_existent.json",
            schema.path());

        CHECK_FALSE(result.has_value());
    }

    TEST_CASE("readJsonFile returns nullopt for non-existent schema")
    {
        const auto config = TempFile("test_config.json");
        config.write(minimalValidConfig);

        const auto result = drip::app::readJsonFile<drip::sim::SimulationConfig>(
            config.path(),
            std::filesystem::temp_directory_path() / "non_existent_schema.json");

        CHECK_FALSE(result.has_value());
    }

    TEST_CASE("readJsonFile returns nullopt for invalid JSON")
    {
        const auto config = TempFile("test_invalid.json");
        const auto schema = TempFile("test_schema.json");

        config.write("{ invalid json }");
        schema.write(minimalValidSchema);

        const auto result = drip::app::readJsonFile<drip::sim::SimulationConfig>(config.path(), schema.path());

        CHECK_FALSE(result.has_value());
    }

    TEST_CASE("readJsonFile parses minimal config with defaults")
    {
        const auto config = TempFile("test_valid_config.json");
        const auto schema = TempFile("test_valid_schema.json");

        config.write(minimalValidConfig);
        schema.write(minimalValidSchema);

        const auto result = drip::app::readJsonFile<drip::sim::SimulationConfig>(config.path(), schema.path());

        REQUIRE(result.has_value());
        CHECK(result->domain.bounds.min.x == doctest::Approx(-1.0));
        CHECK(result->fluid.bounds.max.x == doctest::Approx(1.0));
    }

    TEST_CASE("readJsonFile parses full config")
    {
        const auto config = TempFile("test_full_config.json");
        const auto schema = TempFile("test_full_schema.json");

        config.write(fullConfig);
        schema.write(minimalValidSchema);

        const auto result = drip::app::readJsonFile<drip::sim::SimulationConfig>(config.path(), schema.path());

        REQUIRE(result.has_value());

        SUBCASE("domain bounds are parsed correctly")
        {
            CHECK(result->domain.bounds.min.x == doctest::Approx(-2.0));
            CHECK(result->domain.bounds.min.y == doctest::Approx(-2.0));
            CHECK(result->domain.bounds.min.z == doctest::Approx(-2.0));
            CHECK(result->domain.bounds.max.x == doctest::Approx(2.0));
            CHECK(result->domain.bounds.max.y == doctest::Approx(2.0));
            CHECK(result->domain.bounds.max.z == doctest::Approx(2.0));
        }

        SUBCASE("fluid bounds are parsed correctly")
        {
            CHECK(result->fluid.bounds.min.x == doctest::Approx(-1.0));
            CHECK(result->fluid.bounds.min.y == doctest::Approx(-1.0));
            CHECK(result->fluid.bounds.min.z == doctest::Approx(-1.0));
            CHECK(result->fluid.bounds.max.x == doctest::Approx(1.0));
            CHECK(result->fluid.bounds.max.y == doctest::Approx(1.0));
            CHECK(result->fluid.bounds.max.z == doctest::Approx(1.0));
        }

        SUBCASE("fluid properties are parsed correctly")
        {
            CHECK(result->fluid.properties.spacing == doctest::Approx(0.05));
            CHECK(result->fluid.properties.smoothingRadius == doctest::Approx(0.1));
            CHECK(result->fluid.properties.density == doctest::Approx(998.0));
            CHECK(result->fluid.properties.surfaceTension == doctest::Approx(0.5));
            CHECK(result->fluid.properties.viscosity == doctest::Approx(0.2));
            CHECK(result->fluid.properties.maxVelocity == doctest::Approx(15.0));
            CHECK(result->fluid.properties.speedOfSound == doctest::Approx(60.0));
        }

        SUBCASE("environment gravity is parsed correctly")
        {
            CHECK(result->environment.gravity.x == doctest::Approx(0.0));
            CHECK(result->environment.gravity.y == doctest::Approx(-10.0));
            CHECK(result->environment.gravity.z == doctest::Approx(0.0));
        }
    }

    TEST_CASE("writeJsonFile creates file with valid JSON")
    {
        const auto file = TempFile("test_write_output.json");

        drip::sim::SimulationConfig config;
        config.environment.gravity = {0.0F, -5.0F, 0.0F};

        const auto success = drip::app::writeJsonFile(file.path(), config);

        REQUIRE(success);

        std::ifstream stream(file.path());
        const auto json = nlohmann::json::parse(stream);

        CHECK(json["environment"]["gravity"]["y"] == -5.0F);
    }

    TEST_CASE("writeJsonFile returns false for invalid path")
    {
        auto invalidPath = std::filesystem::path("/nonexistent/directory/file.json");

        const auto config = drip::sim::SimulationConfig {};
        const auto success = drip::app::writeJsonFile(invalidPath, config);

        CHECK_FALSE(success);
    }
}