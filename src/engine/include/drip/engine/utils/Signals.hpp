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
using GameLoopIterationStarted = common::signal::Signal<>;

inline auto frameBufferResized = FrameBufferResized {};
inline auto gameLoopIterationStarted = GameLoopIterationStarted {};

}
