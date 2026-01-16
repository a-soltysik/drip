#pragma once
#include <drip/common/log/LogMessageBuilder.hpp>
#include <exception>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <optional>

#include "Serializers.hpp"  //NOLINT(misc-include-cleaner)

namespace drip::app
{

template <typename T>
auto readJsonFile(const std::filesystem::path& path) -> std::optional<T>
{
    try
    {
        auto file = std::ifstream {path};
        if (!file.is_open())
        {
            common::log::Error("Failed to open configuration file: {}", path.string());
            return {};
        }

        auto jsonFile = nlohmann::json {};
        file >> jsonFile;

        return jsonFile.get<T>();
    }
    catch (const nlohmann::json::exception& e)
    {
        common::log::Error("JSON parsing error, file {}", path.string()).withException(e);
        return {};
    }
    catch (const std::exception& e)
    {
        common::log::Error("Error loading json file {}", path.string()).withException(e);
        return {};
    }
}

template <typename T>
auto writeJsonFile(const std::filesystem::path& path, T&& object) -> bool
{
    try
    {
        auto file = std::ofstream {path};
        if (!file.is_open())
        {
            common::log::Error("Failed to open configuration file: {}", path.string());
            return {};
        }

        file << nlohmann::json(std::forward<T>(object));

        return true;
    }
    catch (const nlohmann::json::exception& e)
    {
        common::log::Error("JSON parsing error, file: {}", path.string()).withException(e);
        return false;
    }
    catch (const std::exception& e)
    {
        common::log::Error("Error writing json file {}", path.string()).withException(e);
        return false;
    }
}

}
