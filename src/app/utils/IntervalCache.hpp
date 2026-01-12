#pragma once
#include <chrono>

namespace drip::app::utils
{

template <typename F, typename T = std::invoke_result_t<F>>
class IntervalCache
{
public:
    IntervalCache(std::chrono::milliseconds interval, F function)
        : _interval(interval),
          _function(function)
    {
    }

    auto get() -> T
    {
        const auto now = std::chrono::steady_clock::now();

        if (now - _lastUpdate >= _interval)
        {
            _lastValue = _function();
            _lastUpdate = now;
        }
        return _lastValue;
    }

private:
    std::chrono::milliseconds _interval;
    std::chrono::steady_clock::time_point _lastUpdate = std::chrono::steady_clock::now();
    F _function;
    T _lastValue = _function();
};

template <typename F>
IntervalCache(std::chrono::milliseconds, F) -> IntervalCache<F>;

}
