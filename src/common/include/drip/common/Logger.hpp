#pragma once

#include <fmt/base.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/os.h>

#include <atomic>
#include <boost/exception/detail/exception_ptr.hpp>
#include <boost/exception/diagnostic_information.hpp>
#include <boost/exception_ptr.hpp>
#include <boost/stacktrace/stacktrace.hpp>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "drip/common/utils/format/ExceptionFormatter.hpp"   // NOLINT(misc-include-cleaner)
#include "drip/common/utils/format/StacktraceFormatter.hpp"  // NOLINT(misc-include-cleaner)

namespace drip::common::log
{

enum class Level : uint8_t
{
    Debug,
    Info,
    Warning,
    Error
};

struct LogEntry
{
    uint32_t index;
    std::string message;
    std::source_location location;
    std::chrono::system_clock::time_point time;
    Level level;
};

class LogSink
{
public:
    LogSink() = default;
    LogSink(const LogSink&) = delete;
    LogSink(LogSink&&) = delete;
    auto operator=(const LogSink&) = delete;
    auto operator=(LogSink&&) = delete;
    virtual ~LogSink() = default;

    virtual void write(const LogEntry& entry) = 0;
    virtual void flush() = 0;
};

class ConsoleSink final : public LogSink
{
public:
    ConsoleSink() = default;

    void write(const LogEntry& entry) override;
    void flush() override;
};

class FileSink final : public LogSink
{
public:
    explicit FileSink(std::string_view logDirectory = "logs");
    ~FileSink() override;

    FileSink(const FileSink&) = delete;
    auto operator=(const FileSink&) -> FileSink& = delete;
    FileSink(FileSink&&) = delete;
    auto operator=(FileSink&&) -> FileSink& = delete;

    void write(const LogEntry& entry) override;
    void flush() override;

private:
    auto openFile() -> bool;

    std::string _logDirectory;
    std::unique_ptr<fmt::ostream> _file;
    std::vector<LogEntry> _buffer;
    bool _isHealthy = false;
};

template <typename... Args>
struct Debug
{
    explicit Debug(std::string_view format,
                   Args&&... args,
                   std::source_location location = std::source_location::current()) noexcept;
};

template <typename... Args>
struct Info
{
    explicit Info(std::string_view format,
                  Args&&... args,
                  std::source_location location = std::source_location::current()) noexcept;
};

template <typename... Args>
struct Warning
{
    explicit Warning(std::string_view format,
                     Args&&... args,
                     std::source_location location = std::source_location::current()) noexcept;
};

template <typename... Args>
struct Error
{
    explicit Error(std::string_view format,
                   Args&&... args,
                   std::source_location location = std::source_location::current()) noexcept;
};

template <typename... Args>
struct Exception
{
    explicit Exception(
        std::string_view format,
        Args&&... args,
        boost::exception_ptr exception = boost::current_exception(),
        boost::stacktrace::stacktrace stacktrace = boost::stacktrace::stacktrace::from_current_exception(),
        std::source_location location = std::source_location::current()) noexcept;
};

namespace internal
{
struct LogDispatcher;
}

class Logger
{
public:
    class Builder;

    Logger(const Logger&) = delete;
    auto operator=(const Logger&) = delete;
    Logger(Logger&&) = delete;
    auto operator=(Logger&&) = delete;
    ~Logger();

    void setLevels(std::span<const Level> newLevels);
    void start();
    void stop();
    [[nodiscard]] auto isRunning() const -> bool;

    template <typename T, typename... Args>
    void addSink(Args&&... args)
    {
        const auto lock = std::scoped_lock(_mutex);
        _sinks.push_back(std::make_unique<T>(std::forward<Args>(args)...));
    }

    void clearSinks();

    void setAutoFlush(bool enabled, std::chrono::milliseconds interval);
    void setFlushOnLevel(Level level);
    void setExceptionHandler(std::function<void(const std::optional<std::exception>&)> exceptionHandler);
    void raiseException(const std::optional<std::exception>& exception) const;

    static auto instance() -> Logger&;

private:
    friend struct internal::LogDispatcher;

    Logger() = default;
    void log(Level level, std::string_view message, const std::source_location& location);
    void flush();
    void flushWorker();
    [[nodiscard]] auto shouldFlushImmediately(Level level) const -> bool;

    std::set<Level> _levels = {Level::Debug, Level::Info, Level::Warning, Level::Error};
    std::vector<std::unique_ptr<LogSink>> _sinks;
    uint32_t _currentIndex = 0;

    std::atomic<bool> _isRunning = false;
    std::atomic<bool> _autoFlushEnabled = false;
    std::chrono::milliseconds _autoFlushInterval {5000};
    Level _immediateFlushLevel = Level::Error;

    std::mutex _mutex;
    std::thread _flushThread;
    std::condition_variable _flushCondition;
    std::atomic<bool> _stopFlushThread = false;
    std::function<void(const std::optional<std::exception>&)> _exceptionHandler = [](const auto&) {
    };
};

namespace internal
{

struct LogDispatcher
{
    static void log(Level level, std::string_view message, const std::source_location& location) noexcept;
};

}

template <typename... Args>
Debug<Args...>::Debug(std::string_view format, Args&&... args, std::source_location location) noexcept
{
    internal::LogDispatcher::log(Level::Debug,
                                 fmt::format(fmt::runtime(format), std::forward<Args>(args)...),
                                 location);
}

template <typename... Args>
Info<Args...>::Info(std::string_view format, Args&&... args, std::source_location location) noexcept
{
    internal::LogDispatcher::log(Level::Info, fmt::format(fmt::runtime(format), std::forward<Args>(args)...), location);
}

template <typename... Args>
Warning<Args...>::Warning(std::string_view format, Args&&... args, std::source_location location) noexcept
{
    internal::LogDispatcher::log(Level::Warning,
                                 fmt::format(fmt::runtime(format), std::forward<Args>(args)...),
                                 location);
}

template <typename... Args>
Error<Args...>::Error(std::string_view format, Args&&... args, std::source_location location) noexcept
{
    internal::LogDispatcher::log(Level::Error,
                                 fmt::format(fmt::runtime(format), std::forward<Args>(args)...),
                                 location);
}

template <typename... Args>
Exception<Args...>::Exception(std::string_view format,
                              Args&&... args,
                              boost::exception_ptr exception,
                              boost::stacktrace::stacktrace stacktrace,
                              std::source_location location) noexcept
{
    internal::LogDispatcher::log(Level::Error,
                                 fmt::format("{}\n{}",
                                             fmt::format(fmt::runtime(format), std::forward<Args>(args)...),
                                             fmt::format("{}\nAt:\n{}", exception, stacktrace)),
                                 location);
}

template <typename... Args>
Debug(std::string_view, Args&&...) -> Debug<Args...>;

template <typename... Args>
Info(std::string_view, Args&&...) -> Info<Args...>;

template <typename... Args>
Warning(std::string_view, Args&&...) -> Warning<Args...>;

template <typename... Args>
Error(std::string_view, Args&&...) -> Error<Args...>;

template <typename... Args>
Exception(std::string_view, Args&&...) -> Exception<Args...>;
}