#include "drip/common/Logger.hpp"

#include <fmt/base.h>
#include <fmt/chrono.h>  // NOLINT(misc-include-cleaner)
#include <fmt/format.h>
#include <fmt/os.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace drip::common::log
{

namespace
{

[[nodiscard]] constexpr auto getLevelTag(Level level) -> std::string_view
{
    using namespace std::string_view_literals;
    switch (level)
    {
    case Level::Debug:
        return "[DBG]"sv;
    case Level::Info:
        return "[INF]"sv;
    case Level::Warning:
        return "[WRN]"sv;
    case Level::Error:
        return "[ERR]"sv;
    default:
        [[unlikely]] return "[???]"sv;
    }
}

[[nodiscard]] constexpr auto getFunctionName(std::string_view function) -> std::string_view
{
    const auto nameWithReturnType = function.substr(0, function.find('('));
    return nameWithReturnType.substr(nameWithReturnType.rfind(' ') + 1);
}

[[nodiscard]] auto formatLogEntry(const LogEntry& entry) -> std::string
{
    return fmt::format("{:%H:%M:%S} {} {}:{}, {}",
                       entry.time,
                       getLevelTag(entry.level),
                       getFunctionName(entry.location.function_name()),
                       entry.location.line(),
                       entry.message);
}

}

void ConsoleSink::write(const LogEntry& entry)
{
    fmt::println("{}", formatLogEntry(entry));
}

void ConsoleSink::flush() { }

FileSink::FileSink(std::string_view logDirectory)
    : _logDirectory(logDirectory)
{
    openFile();
}

FileSink::~FileSink()
{
    try
    {
        flush();
        if (_file)
        {
            _file->close();
        }
    }
    catch (const std::exception& ex)
    {
        Logger::instance().raiseException(ex);
    }
    catch (...)
    {
        Logger::instance().raiseException({});
    }
}

auto FileSink::openFile() -> bool
{
    try
    {
        std::filesystem::create_directories(_logDirectory);

        const auto time = std::chrono::system_clock::now();
        const auto filename =
            fmt::format("{}/{:%F_%H_%M_%S}.log", _logDirectory, std::chrono::floor<std::chrono::seconds>(time));

        _file = std::make_unique<fmt::ostream>(fmt::output_file(filename));
        _isHealthy = true;
        return true;
    }
    catch (const std::system_error&)
    {
        _isHealthy = false;
        return false;
    }
}

void FileSink::write(const LogEntry& entry)
{
    if (!_isHealthy)
    {
        return;
    }

    _buffer.push_back(entry);
}

void FileSink::flush()
{
    if (!_isHealthy || !_file)
    {
        return;
    }

    try
    {
        for (const auto& entry : _buffer)
        {
            _file->print("{}\n", formatLogEntry(entry));
        }
        _buffer.clear();
    }
    catch (const std::exception& ex)
    {
        _isHealthy = false;
        Logger::instance().raiseException(ex);
    }
    catch (...)
    {
        _isHealthy = false;
        Logger::instance().raiseException({});
    }
}

namespace internal
{

void LogDispatcher::log(Level level, std::string_view message, const std::source_location& location) noexcept
{
    try
    {
        Logger::instance().log(level, message, location);
    }
    catch (const std::exception& ex)
    {
        Logger::instance().raiseException(ex);
    }
    catch (...)
    {
        Logger::instance().raiseException({});
    }
}

}

auto Logger::instance() -> Logger&
{
    static Logger logger;
    return logger;
}

void Logger::log(Level level, std::string_view message, const std::source_location& location)
{
    if (!_isRunning.load(std::memory_order_acquire))
    {
        return;
    }

    if (!_levels.contains(level))
    {
        return;
    }

    const LogEntry entry {.index = _currentIndex++,
                          .message = std::string {message},
                          .location = location,
                          .time = std::chrono::system_clock::now(),
                          .level = level};

    {
        const auto lock = std::scoped_lock(_mutex);
        for (const auto& sink : _sinks)
        {
            sink->write(entry);
        }
    }

    if (shouldFlushImmediately(level))
    {
        flush();
    }
}

void Logger::setLevels(std::span<const Level> newLevels)
{
    const auto lock = std::scoped_lock(_mutex);
    _levels = {newLevels.begin(), newLevels.end()};
}

void Logger::start()
{
    if (_isRunning.load(std::memory_order_acquire))
    {
        return;
    }

    _isRunning.store(true, std::memory_order_release);

    if (_autoFlushEnabled.load(std::memory_order_acquire))
    {
        _stopFlushThread.store(false, std::memory_order_release);
        _flushThread = std::thread([this] {
            flushWorker();
        });
    }
}

void Logger::stop()
{
    if (!_isRunning.load(std::memory_order_acquire))
    {
        return;
    }

    _isRunning.store(false, std::memory_order_release);

    if (_flushThread.joinable())
    {
        _stopFlushThread.store(true, std::memory_order_release);
        _flushCondition.notify_one();
        _flushThread.join();
    }
}

auto Logger::isRunning() const -> bool
{
    return _isRunning.load(std::memory_order_acquire);
}

void Logger::clearSinks()
{
    const auto lock = std::scoped_lock(_mutex);
    _sinks.clear();
}

void Logger::raiseException(const std::optional<std::exception>& exception) const
{
    _exceptionHandler(exception);
}

void Logger::setAutoFlush(bool enabled, std::chrono::milliseconds interval)
{
    const bool wasEnabled = _autoFlushEnabled.load(std::memory_order_acquire);

    _autoFlushEnabled.store(enabled, std::memory_order_release);
    _autoFlushInterval = interval;

    if (!wasEnabled && enabled && _isRunning.load(std::memory_order_acquire))
    {
        _stopFlushThread.store(false, std::memory_order_release);
        _flushThread = std::thread([this] {
            flushWorker();
        });
    }
    else if (wasEnabled && !enabled && _flushThread.joinable())
    {
        _stopFlushThread.store(true, std::memory_order_release);
        _flushCondition.notify_one();
        _flushThread.join();
    }
}

void Logger::setFlushOnLevel(Level level)
{
    _immediateFlushLevel = level;
}

void Logger::setExceptionHandler(std::function<void(const std::optional<std::exception>&)> exceptionHandler)
{
    const auto lock = std::scoped_lock(_mutex);
    _exceptionHandler = std::move(exceptionHandler);
}

void Logger::flush()
{
    const auto lock = std::scoped_lock(_mutex);
    for (const auto& sink : _sinks)
    {
        sink->flush();
    }
}

void Logger::flushWorker()
{
    while (!_stopFlushThread.load(std::memory_order_acquire))
    {
        auto lock = std::unique_lock(_mutex);
        _flushCondition.wait_for(lock, _autoFlushInterval, [this] {
            return _stopFlushThread.load(std::memory_order_acquire);
        });

        for (const auto& sink : _sinks)
        {
            sink->flush();
        }
    }
}

auto Logger::shouldFlushImmediately(Level level) const -> bool
{
    return static_cast<uint8_t>(level) >= static_cast<uint8_t>(_immediateFlushLevel);
}

Logger::~Logger()
{
    stop();
}

}
