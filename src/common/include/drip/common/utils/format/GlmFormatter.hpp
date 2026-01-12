#pragma once

#include <fmt/base.h>

#include <glm/detail/qualifier.hpp>
#include <glm/gtx/string_cast.hpp>
#include <string_view>

template <glm::length_t L, typename T, glm::qualifier Q>
struct fmt::formatter<glm::vec<L, T, Q>> : formatter<std::string_view>
{
    template <typename FormatContext>
    [[nodiscard]] auto format(glm::vec<L, T, Q> vec, FormatContext& ctx) const -> decltype(ctx.out())
    {
        return formatter<std::string_view>::format(glm::to_string(vec), ctx);
    }
};
