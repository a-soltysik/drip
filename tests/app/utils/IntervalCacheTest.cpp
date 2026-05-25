//NOLINTBEGIN(misc-include-cleaner) gcc requires bits/chrono which is not standard compliant

#include <doctest/doctest.h>

#include <chrono>
#include <string>
#include <thread>

#include "utils/IntervalCache.hpp"

using namespace std::chrono_literals;

TEST_SUITE("IntervalCache")
{
    TEST_CASE("returns initial value without waiting")
    {
        int callCount = 0;
        auto cache = drip::app::utils::IntervalCache(100ms, [&callCount]() {
            callCount++;
            return 42;
        });

        CHECK(cache.get() == 42);
        CHECK(callCount == 1);
    }

    TEST_CASE("returns cached value within interval")
    {
        int callCount = 0;
        auto cache = drip::app::utils::IntervalCache(100ms, [&callCount]() {
            callCount++;
            return callCount * 10;
        });

        auto first = cache.get();
        auto second = cache.get();
        auto third = cache.get();

        CHECK(first == 10);
        CHECK(second == 10);
        CHECK(third == 10);
        CHECK(callCount == 1);
    }

    TEST_CASE("refreshes value after interval expires")
    {
        int callCount = 0;
        auto cache = drip::app::utils::IntervalCache(50ms, [&callCount]() {
            callCount++;
            return callCount * 10;
        });

        auto first = cache.get();
        CHECK(first == 10);
        CHECK(callCount == 1);

        std::this_thread::sleep_for(60ms);

        auto second = cache.get();
        CHECK(second == 20);
        CHECK(callCount == 2);
    }

    TEST_CASE("works with different return types")
    {
        SUBCASE("string")
        {
            auto cache = drip::app::utils::IntervalCache(100ms, []() {
                return std::string("hello");
            });
            CHECK(cache.get() == "hello");
        }

        SUBCASE("float")
        {
            auto cache = drip::app::utils::IntervalCache(100ms, []() {
                return 3.14F;
            });
            CHECK(cache.get() == doctest::Approx(3.14));
        }
    }
}

//NOLINTEND(misc-include-cleaner)
