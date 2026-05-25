#pragma once
#include <drip/common/log/LogMessageBuilder.hpp>
#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>

namespace drip::app::utils
{

template <typename T>
auto readFile(const std::filesystem::path& path) -> std::optional<T>
{
    try
    {
        auto file = std::ifstream {path};
        if (!file.is_open())
        {
            common::log::Error("Failed to open configuration file: {}", path.string());
            return {};
        }

        auto jsonFile = T {};
        file >> jsonFile;

        return jsonFile;
    }
    catch (const std::exception& e)
    {
        common::log::Error("Error loading file {}", path.string()).withException(e);
        return {};
    }
}

template <typename T>
auto writeFile(const std::filesystem::path& path, T&& object) -> bool
{
    try
    {
        auto file = std::ofstream {path};
        if (!file.is_open())
        {
            common::log::Error("Failed to open configuration file: {}", path.string());
            return false;
        }

        file << std::forward<T>(object);

        return true;
    }
    catch (const std::exception& e)
    {
        common::log::Error("Error writing json file {}", path.string()).withException(e);
        return false;
    }
}

}
