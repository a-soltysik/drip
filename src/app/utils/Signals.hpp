#pragma once

#include <drip/common/utils/Signal.hpp>
#include <drip/gfx/Window.hpp>

namespace drip::app::signal
{

struct KeyboardStateChangedData
{
    gfx::Window::Id id;
    int key;
    int scancode;
    int action;
    int mods;
};

using KeyboardStateChanged = common::signal::Signal<KeyboardStateChangedData>;

struct MouseButtonStateChangedData
{
    gfx::Window::Id id;
    int button;
    int action;
    int mods;
};

using MouseButtonStateChanged = common::signal::Signal<MouseButtonStateChangedData>;

struct CursorPositionChangedData
{
    gfx::Window::Id id;
    double x;
    double y;
};

using CursorPositionChanged = common::signal::Signal<CursorPositionChangedData>;

using MainLoopIterationStarted = common::signal::Signal<>;

inline auto keyboardStateChanged = KeyboardStateChanged {};
inline auto mouseButtonStateChanged = MouseButtonStateChanged {};
inline auto cursorPositionChanged = CursorPositionChanged {};
inline auto mainLoopIterationStarted = MainLoopIterationStarted {};

}
