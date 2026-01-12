#pragma once

// clang-format off
#include <drip/common/utils/Assert.hpp>
// clang-format on

#include <filesystem>
#include <optional>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_handles.hpp>

namespace drip::gfx
{

class Shader
{
public:
    enum class Type : uint8_t
    {
        Vertex,
        TessellationControl,
        TessellationEvaluation,
        Geometry,
        Fragment,
        Compute
    };

    Shader(const vk::ShaderModule& shaderModule, Type shaderType, const vk::Device& device) noexcept;
    Shader(const Shader&) = delete;
    Shader(Shader&&) = delete;
    auto operator=(const Shader&) -> Shader& = delete;
    auto operator=(Shader&&) -> Shader& = delete;
    ~Shader() noexcept;

    [[nodiscard]] static auto createFromFile(const vk::Device& device, const std::filesystem::path& path)
        -> std::optional<Shader>;
    [[nodiscard]] static auto createFromFile(const vk::Device& device, const std::filesystem::path& path, Type type)
        -> std::optional<Shader>;
    [[nodiscard]] static auto createFromRawData(const vk::Device& device,
                                                const std::vector<uint32_t>& buffer,
                                                Type type) -> std::optional<Shader>;

    [[nodiscard]] static constexpr auto getEntryPointName() -> const char*
    {
        return "main";
    }

    const vk::ShaderModule module;
    const Type type;

private:
    const vk::Device& _device;
};

}
