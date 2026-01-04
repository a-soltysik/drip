// clang-format off
#include <drip/common/utils/Assert.hpp> // NOLINT(misc-include-cleaner)
// clang-format on

#include "Shader.hpp"

#include <cstddef>
#include <cstdint>
#include <drip/common/log/LogMessageBuilder.hpp>
#include <filesystem>
#include <fstream>
#include <ios>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "drip/engine/utils/format/ResultFormatter.hpp"  //NOLINT(misc-include-cleaner)

namespace drip::engine::gfx
{

auto Shader::createFromFile(const vk::Device& device, const std::filesystem::path& path) -> std::optional<Shader>
{
    using namespace std::string_view_literals;
    static const std::unordered_map<std::string_view, Type> extensions = {
        {".vert"sv, Type::Vertex                },
        {".tesc"sv, Type::TessellationControl   },
        {".tese"sv, Type::TessellationEvaluation},
        {".geom"sv, Type::Geometry              },
        {".frag"sv, Type::Fragment              },
        {".comp"sv, Type::Compute               }
    };

    const auto shaderExtension = path.stem().extension().string();

    const auto it = extensions.find(shaderExtension);
    if (it == extensions.cend())
    {
        common::log::Warning("File extension: {} is not supported (filename: {})", shaderExtension, shaderExtension);
        return {};
    }
    return createFromFile(device, path, it->second);
}

auto Shader::createFromFile(const vk::Device& device, const std::filesystem::path& path, Type type)
    -> std::optional<Shader>
{
    auto fin = std::ifstream(path, std::ios::ate | std::ios::binary);

    if (!fin.is_open())
    {
        common::log::Warning("File {} cannot be opened", path.string());
        return {};
    }

    const auto fileSize = fin.tellg();
    const auto bufferSize = static_cast<size_t>(fileSize) / sizeof(uint32_t);
    auto buffer = std::vector<uint32_t>(bufferSize);

    fin.seekg(0);
    fin.read(reinterpret_cast<char*>(buffer.data()), fileSize);

    return createFromRawData(device, buffer, type);
}

auto Shader::createFromRawData(const vk::Device& device, const std::vector<uint32_t>& buffer, Type type)
    -> std::optional<Shader>
{
    const auto createInfo =
        vk::ShaderModuleCreateInfo {.codeSize = buffer.size() * sizeof(uint32_t), .pCode = buffer.data()};
    const auto shaderModuleResult = device.createShaderModule(createInfo);
    if (shaderModuleResult.result == vk::Result::eSuccess)
    {
        return std::make_optional<Shader>(shaderModuleResult.value, type, device);
    }

    common::log::Warning("Creating shader module didn't succeed: {}", shaderModuleResult.result);
    return {};
}

Shader::~Shader() noexcept
{
    common::log::Info("Destroying shader [{}]", static_cast<void*>(module));
    _device.destroy(module);
}

Shader::Shader(const vk::ShaderModule& shaderModule, Type shaderType, const vk::Device& device) noexcept
    : module {shaderModule},
      type {shaderType},
      _device {device}
{
    common::log::Info("Created shader [{}]", static_cast<void*>(module));
}

}
