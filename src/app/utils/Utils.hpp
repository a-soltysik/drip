#pragma once

#include <optional>
#include <utility>

namespace drip::app::utils
{

template <typename T, typename U>
auto and_both(std::optional<T> lhs, std::optional<U> rhs) -> std::optional<std::pair<T, U>>
{
    if (lhs && rhs)
    {
        return std::make_pair(std::move(*lhs), std::move(*rhs));
    }
    return std::nullopt;
}

}
