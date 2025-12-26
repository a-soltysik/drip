#pragma once

// clang-format off
#include <drip/common/utils/Assert.hpp>
// clang-format on

#include <memory>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "drip/engine/rendering/system/RenderSystem.hpp"
#include "drip/engine/resource/Object.hpp"
#include "vulkan/memory/Descriptor.hpp"
#include "vulkan/pipeline/Pipeline.hpp"

namespace drip::engine::gfx
{

class Device;
struct FrameInfo;

class DefaultRenderSystem : public RenderSystem
{
public:
    DefaultRenderSystem(const Device& device, vk::RenderPass renderPass);
    DefaultRenderSystem(const DefaultRenderSystem&) = delete;
    DefaultRenderSystem(DefaultRenderSystem&&) = delete;
    auto operator=(const DefaultRenderSystem&) = delete;
    auto operator=(DefaultRenderSystem&&) = delete;
    ~DefaultRenderSystem() noexcept override;

    void render(const FrameInfo& frameInfo) const override;

private:
    static auto createPipelineLayout(const Device& device, vk::DescriptorSetLayout setLayout) -> vk::PipelineLayout;
    static auto createPipeline(const Device& device, vk::RenderPass renderPass, vk::PipelineLayout pipelineLayout)
        -> std::unique_ptr<Pipeline>;

    void renderObject(const Object& object, const FrameInfo& frameInfo) const;

    const Device& _device;
    std::unique_ptr<DescriptorSetLayout> _descriptorLayout;
    vk::PipelineLayout _pipelineLayout;
    std::unique_ptr<Pipeline> _pipeline;
};

}
