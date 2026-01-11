#pragma once

#include <drip/common/utils/Signal.hpp>
#include <drip/engine/Window.hpp>

namespace drip::app::signal
{

struct KeyboardStateChangedData
{
    engine::Window::Id id;
    int key;
    int scancode;
    int action;
    int mods;
};

using KeyboardStateChanged = common::signal::Signal<KeyboardStateChangedData>;

struct MouseButtonStateChangedData
{
    engine::Window::Id id;
    int button;
    int action;
    int mods;
};

using MouseButtonStateChanged = common::signal::Signal<MouseButtonStateChangedData>;

struct CursorPositionChangedData
{
    engine::Window::Id id;
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
