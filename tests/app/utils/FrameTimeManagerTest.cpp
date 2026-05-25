//NOLINTBEGIN(misc-include-cleaner) gcc requires bits/chrono which is not standard compliant

#include <doctest/doctest.h>

#include <thread>

#include "utils/FrameTimeManager.hpp"

using namespace std::chrono_literals;

TEST_SUITE("FrameTimeManager")
{
    TEST_CASE("initial state")
    {
        const auto manager = drip::app::utils::FrameTimeManager {};

        CHECK(manager.getFrameCount() == 0);
        CHECK(manager.getDelta() == doctest::Approx(0.0));
    }

    TEST_CASE("frame count increments on update")
    {
        auto manager = drip::app::utils::FrameTimeManager {};

        manager.update();
        CHECK(manager.getFrameCount() == 1);

        manager.update();
        CHECK(manager.getFrameCount() == 2);

        manager.update();
        CHECK(manager.getFrameCount() == 3);
    }

    TEST_CASE("delta time is positive after update")
    {
        auto manager = drip::app::utils::FrameTimeManager {};

        std::this_thread::sleep_for(10ms);
        manager.update();

        CHECK(manager.getDelta() > 0.F);
    }

    TEST_CASE("delta time reflects elapsed time")
    {
        auto manager = drip::app::utils::FrameTimeManager {};

        std::this_thread::sleep_for(100ms);
        manager.update();

        CHECK(manager.getDelta() == doctest::Approx(0.1).epsilon(0.025));
    }

    TEST_CASE("mean frame time is calculated correctly")
    {
        auto manager = drip::app::utils::FrameTimeManager {};

        std::this_thread::sleep_for(100ms);
        manager.update();
        std::this_thread::sleep_for(100ms);
        manager.update();

        auto meanTime = manager.getMeanFrameTime();
        CHECK(meanTime == doctest::Approx(0.1).epsilon(0.025));
    }

    TEST_CASE("mean frame rate is inverse of mean frame time")
    {
        auto manager = drip::app::utils::FrameTimeManager {};

        std::this_thread::sleep_for(20ms);
        manager.update();
        std::this_thread::sleep_for(20ms);
        manager.update();

        auto meanTime = manager.getMeanFrameTime();
        auto meanRate = manager.getMeanFrameRate();

        CHECK(meanRate == doctest::Approx(1.0 / static_cast<double>(meanTime)));
    }
}

//NOLINTEND(misc-include-cleaner)
