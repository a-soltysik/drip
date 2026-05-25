#pragma once
#include <drip/common/log/LogMessageBuilder.hpp>
#include <exception>
#include <filesystem>
#include <fstream>
#include <nlohmann/json-schema.hpp>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <optional>

#include "Serializers.hpp"  //NOLINT(misc-include-cleaner)
#include "utils/FileReader.hpp"
#include "utils/Utils.hpp"

namespace drip::app
{

template <typename T>
auto readJsonFile(const std::filesystem::path& path, const std::filesystem::path validatorPath) -> std::optional<T>
{
    const auto tryMakeValidator =
        [&validatorPath](const auto& validatorJson) -> std::optional<nlohmann::json_schema::json_validator> {
        try
        {
            auto validator = nlohmann::json_schema::json_validator {};
            validator.set_root_schema(validatorJson);
            return validator;
        }
        catch (const std::exception& e)
        {
            common::log::Error("JSON schema parsing error, file {}", validatorPath.string()).withException(e);
            return std::nullopt;
        }
    };

    const auto tryValidate = [&path, &validatorPath](const auto& pair) -> std::optional<nlohmann::json> {
        const auto& [config, validator] = pair;
        try
        {
            validator.validate(config);
            return config;
        }
        catch (const std::exception& e)
        {
            common::log::Error("JSON validation error, file {}, schema {}", path.string(), validatorPath.string())
                .withException(e);
            return std::nullopt;
        }
    };

    const auto tryParse = [&path](const auto& configJson) -> std::optional<T> {
        try
        {
            return configJson.template get<T>();
        }
        catch (const nlohmann::json::exception& e)
        {
            common::log::Error("JSON parsing error, file {}", path.string()).withException(e);
            return std::nullopt;
        }
    };

    return utils::and_both(utils::readFile<nlohmann::json>(path),
                           utils::readFile<nlohmann::json>(validatorPath).and_then(tryMakeValidator))
        .and_then(tryValidate)
        .and_then(tryParse);
}

template <typename T>
auto writeJsonFile(const std::filesystem::path& path, T&& object) -> bool
{
    try
    {
        return utils::writeFile(path, nlohmann::json(std::forward<T>(object)));
    }
    catch (const nlohmann::json::exception& e)
    {
        common::log::Error("JSON parsing error, file: {}", path.string()).withException(e);
        return false;
    }
}
}
