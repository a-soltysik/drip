#pragma once

#include <fmt/base.h>
#include <fmt/format.h>

#include <boost/exception/detail/exception_ptr.hpp>
#include <boost/stacktrace/stacktrace.hpp>
#include <source_location>
#include <string_view>

namespace drip::common::log
{

enum class Level : uint8_t
{
    Debug,
    Info,
    Warning,
    Error
};

namespace detail
{

class LogMessageBuilder
{
public:
    explicit LogMessageBuilder(std::string message, const std::source_location& location, Level level);
    LogMessageBuilder(const LogMessageBuilder&) = delete;
    LogMessageBuilder(LogMessageBuilder&&) = delete;
    auto operator=(const LogMessageBuilder&) = delete;
    auto operator=(LogMessageBuilder&&) = delete;
    ~LogMessageBuilder();

    template <typename T>
    auto withException(const T& exception) -> LogMessageBuilder&
    {
        return withStacktraceFromCurrentException(boost::make_exception_ptr(exception));
    }

    auto withCurrentException(boost::exception_ptr exception = boost::current_exception()) -> LogMessageBuilder&;
    auto withStacktrace(boost::stacktrace::stacktrace stacktrace = boost::stacktrace::stacktrace {})
        -> LogMessageBuilder&;
    auto withStacktraceFromCurrentException(
        boost::stacktrace::stacktrace = boost::stacktrace::stacktrace::from_current_exception()) -> LogMessageBuilder&;
    auto asString() -> std::string;

private:
    void log() noexcept;

    std::string _message;
    std::source_location _location;
    Level _level;
    bool _asString = false;
};

}

template <typename... Args>
struct Debug : detail::LogMessageBuilder
{
    explicit Debug(std::string_view format,
                   Args&&... args,
                   const std::source_location& location = std::source_location::current()) noexcept;
};

template <typename... Args>
struct Info : detail::LogMessageBuilder
{
    explicit Info(std::string_view format,
                  Args&&... args,
                  const std::source_location& location = std::source_location::current()) noexcept;
};

template <typename... Args>
struct Warning : detail::LogMessageBuilder
{
    explicit Warning(std::string_view format,
                     Args&&... args,
                     const std::source_location& location = std::source_location::current()) noexcept;
};

template <typename... Args>
struct Error : detail::LogMessageBuilder
{
    explicit Error(std::string_view format,
                   Args&&... args,
                   const std::source_location& location = std::source_location::current()) noexcept;
};

template <typename... Args>
Debug<Args...>::Debug(std::string_view format, Args&&... args, const std::source_location& location) noexcept
    : LogMessageBuilder(fmt::format(fmt::runtime(format), std::forward<Args>(args)...), location, Level::Debug)
{
}

template <typename... Args>
Info<Args...>::Info(std::string_view format, Args&&... args, const std::source_location& location) noexcept
    : LogMessageBuilder(fmt::format(fmt::runtime(format), std::forward<Args>(args)...), location, Level::Info)
{
}

template <typename... Args>
Warning<Args...>::Warning(std::string_view format, Args&&... args, const std::source_location& location) noexcept
    : LogMessageBuilder(fmt::format(fmt::runtime(format), std::forward<Args>(args)...), location, Level::Warning)
{
}

template <typename... Args>
Error<Args...>::Error(std::string_view format, Args&&... args, const std::source_location& location) noexcept
    : LogMessageBuilder(fmt::format(fmt::runtime(format), std::forward<Args>(args)...), location, Level::Error)
{
}

template <typename... Args>
Debug(std::string_view, Args&&...) -> Debug<Args...>;

template <typename... Args>
Info(std::string_view, Args&&...) -> Info<Args...>;

template <typename... Args>
Warning(std::string_view, Args&&...) -> Warning<Args...>;

template <typename... Args>
Error(std::string_view, Args&&...) -> Error<Args...>;

}