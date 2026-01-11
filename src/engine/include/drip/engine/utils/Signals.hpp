#pragma once

#include <drip/common/utils/Signal.hpp>

#include "drip/engine/Window.hpp"

namespace drip::engine::signal
{
struct FrameBufferResizedData
{
    Window::Id id;
    int x;
    int y;
};

using FrameBufferResized = common::signal::Signal<FrameBufferResizedData>;
inline auto frameBufferResized = FrameBufferResized {};

}
