#pragma once

// clang-format off
#include <drip/common/utils/Assert.hpp>
// clang-format on

#include <drip/common/utils/Signal.hpp>
#include <functional>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "drip/engine/Window.hpp"
#include "drip/engine/scene/Scene.hpp"

namespace drip::engine::signal
{
struct FrameBufferResizedData
{
    Window::Id id;
    int x;
    int y;
};

using FrameBufferResized = common::signal::Signal<FrameBufferResizedData>;

struct BeginGuiRenderData
{
    vk::CommandBuffer commandBuffer;
    std::reference_wrapper<gfx::Scene> scene;
};

using BeginGuiRender = common::signal::Signal<BeginGuiRenderData>;

using GameLoopIterationStarted = common::signal::Signal<>;

inline auto frameBufferResized = FrameBufferResized {};
inline auto beginGuiRender = BeginGuiRender {};
inline auto gameLoopIterationStarted = GameLoopIterationStarted {};

}