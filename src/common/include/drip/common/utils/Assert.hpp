#pragma once

#include <fmt/base.h>
#include <fmt/format.h>

#include <concepts>
#include <cstdlib>
#include <optional>
#include <source_location>
#include <string_view>
#include <type_traits>
#include <utility>

#include "drip/common/log/LogMessageBuilder.hpp"

namespace drip::common
{

template <typename T>
concept Result = requires(T result) {
    { result.value };
    { result.result } -> std::equality_comparable;
};

template <Result T>
struct ResultHelper
{
    using Ok = decltype(std::declval<T>().value);
    using Error = decltype(std::declval<T>().result);
};

[[noreturn]] inline auto panic() noexcept
{
    std::abort();
}

[[noreturn]] inline auto panic(std::string_view message,
                               std::source_location location = std::source_location::current()) noexcept
{
    log::detail::LogMessageBuilder {fmt::format("Terminal error: {}", message), location, log::Level::Error};
    panic();
}

template <typename T>
auto expect(T&& result,
            const std::equality_comparable_with<T> auto& expected,
            std::string_view message,
            std::source_location location = std::source_location::current()) noexcept -> T
{
    if (result != expected) [[unlikely]]
    {
        if constexpr (fmt::is_formattable<std::decay_t<T>>() || std::is_enum<std::decay_t<T>>())
        {
            panic(fmt::format("{}: {}", message, result), location);
        }
        else
        {
            panic(message, location);
        }
    }
    return std::forward<T>(result);
}

template <Result T>
auto expect(T&& result,
            const typename ResultHelper<T>::Error& expected,
            std::string_view message,
            std::source_location location = std::source_location::current()) noexcept
    -> ResultHelper<std::remove_reference_t<T>>::Ok
{
    if (result.result != expected) [[unlikely]]
    {
        panic(fmt::format("{}: {}", message, result.result), location);
    }
    return std::forward<T>(result).value;
}

template <typename T>
auto expect(T&& value,
            std::invocable<const std::remove_reference_t<T>&> auto predicate,
            std::string_view message,
            std::source_location location = std::source_location::current()) noexcept -> T
{
    if (!predicate(value)) [[unlikely]]
    {
        panic(message, location);
    }
    return std::forward<T>(value);
}

void expect(std::convertible_to<bool> auto&& result,
            std::string_view message,
            std::source_location location = std::source_location::current()) noexcept
{
    if (!result) [[unlikely]]
    {
        panic(message, location);
    }
}

void expectNot(std::convertible_to<bool> auto&& result,
               std::string_view message,
               std::source_location location = std::source_location::current()) noexcept
{
    if (result) [[unlikely]]
    {
        panic(message, location);
    }
}

template <typename T>
auto expectNot(T&& result,
               const std::equality_comparable_with<T> auto& notExpected,
               std::string_view message,
               std::source_location location = std::source_location::current()) noexcept -> T
{
    if (result == notExpected) [[unlikely]]
    {
        if constexpr (fmt::is_formattable<std::decay_t<T>>() || std::is_enum<std::decay_t<T>>())
        {
            panic(fmt::format("{}: {}", message, result), location);
        }
        else
        {
            panic(message, location);
        }
    }
    return std::forward<T>(result);
}

template <Result T>
auto expectNot(T&& result,
               const typename ResultHelper<T>::Error& notExpected,
               std::string_view message,
               std::source_location location = std::source_location::current()) noexcept
    -> ResultHelper<std::remove_reference_t<T>>::Ok
{
    if (result.result == notExpected) [[unlikely]]
    {
        panic(fmt::format("{}: {}", message, result.result), location);
    }
    return std::forward<T>(result).value;
}

template <typename T>
auto expectNot(T&& value,
               std::invocable<const std::remove_reference_t<T>&> auto predicate,
               std::string_view message,
               std::source_location location = std::source_location::current()) noexcept -> T
{
    if (predicate(value)) [[unlikely]]
    {
        panic(message, location);
    }
    return std::forward<T>(value);
}

template <typename T>
auto expect(std::optional<T> value,
            std::string_view message,
            std::source_location location = std::source_location::current()) noexcept -> T
{
    if (!value.has_value()) [[unlikely]]
    {
        panic(message, location);
    }
    return std::move(value).value();
}

template <typename T>
auto shouldBe(const T& result,
              const std::equality_comparable_with<T> auto& expected,
              std::string_view message,
              std::source_location location = std::source_location::current()) noexcept -> bool
{
    if (result != expected) [[unlikely]]
    {
        if constexpr (fmt::is_formattable<std::decay_t<T>>() || std::is_enum<std::decay_t<T>>())
        {
            log::detail::LogMessageBuilder {fmt::format("{}: {}", message, result), location, log::Level::Error};
        }
        else
        {
            log::detail::LogMessageBuilder {fmt::format("{}", message), location, log::Level::Error};
        }
        return false;
    }
    return true;
}

auto shouldBe(std::convertible_to<bool> auto&& result,
              std::string_view message,
              std::source_location location = std::source_location::current()) noexcept -> bool
{
    if (!result) [[unlikely]]
    {
        log::detail::LogMessageBuilder {fmt::format("{}", message), location, log::Level::Warning};
        return false;
    }
    return true;
}

template <typename T>
auto shouldBe(const T& value,
              std::invocable<const std::remove_reference_t<T>&> auto predicate,
              std::string_view message,
              std::source_location location = std::source_location::current()) noexcept -> bool
{
    if (!predicate(value)) [[unlikely]]
    {
        log::detail::LogMessageBuilder {fmt::format("{}", message), location, log::Level::Warning};
        return false;
    }
    return true;
}

template <typename T>
auto shouldNotBe(const T& result,
                 const std::equality_comparable_with<T> auto& notExpected,
                 std::string_view message,
                 std::source_location location = std::source_location::current()) noexcept -> bool
{
    if (result == notExpected) [[unlikely]]
    {
        if constexpr (fmt::is_formattable<std::decay_t<T>>() || std::is_enum<std::decay_t<T>>())
        {
            log::detail::LogMessageBuilder {fmt::format("{}: {}", message, result), location, log::Level::Warning};
        }
        else
        {
            log::detail::LogMessageBuilder {fmt::format("{}", message), location, log::Level::Warning};
        }
        return false;
    }
    return true;
}

}

#if NDEBUG
#    define DRIP_DEBUG_EXPECT(condition) ((void) 0)
#    define DRIP_EXPECT(condition) drip::common::shouldBe(condition, #condition)
#else
#    define DRIP_DEBUG_EXPECT(condition) drip::common::expect(condition, #condition)
#    define DRIP_EXPECT(condition) drip::common::expect(condition, #condition)
#endif
