#pragma once
#include <drip/common/log/LogMessageBuilder.hpp>
#include <exception>
#include <filesystem>
#include <nlohmann/json-schema.hpp>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <string_view>

#include "Serializers.hpp"  //NOLINT(misc-include-cleaner)
#include "utils/FileReader.hpp"

namespace drip::app
{

template <typename T>
auto readJsonFile(const std::filesystem::path& path, std::string_view schema) -> std::optional<T>
{
    const auto tryValidate = [&path, schema](const auto& json) -> std::optional<nlohmann::json> {
        try
        {
            auto validator = nlohmann::json_schema::json_validator {};
            validator.set_root_schema(nlohmann::json::parse(schema));
            validator.validate(json);
            return json;
        }
        catch (const std::exception& e)
        {
            static constexpr auto schemaPreviewSize = 100;
            common::log::Error("JSON validation error, file {}, schema {}",
                               path.string(),
                               schema.substr(0, schemaPreviewSize))
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

    return utils::readFile<nlohmann::json>(path).and_then(tryValidate).and_then(tryParse);
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
