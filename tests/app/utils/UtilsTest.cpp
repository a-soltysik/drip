#include <doctest/doctest.h>

#include <optional>
#include <string>
#include <utility>

#include "utils/Utils.hpp"

TEST_SUITE("and_both")
{
    TEST_CASE("returns pair when both optionals have values")
    {
        auto a = std::optional<int>(42);
        auto b = std::optional<std::string>("hello");

        auto result = drip::app::utils::and_both(a, b);

        REQUIRE(result.has_value());
        CHECK(result->first == 42);
        CHECK(result->second == "hello");
    }

    TEST_CASE("returns nullopt when first optional is empty")
    {
        auto a = std::optional<int>();
        auto b = std::optional<std::string>("hello");

        auto result = drip::app::utils::and_both(a, b);

        CHECK_FALSE(result.has_value());
    }

    TEST_CASE("returns nullopt when second optional is empty")
    {
        auto a = std::optional<int>(42);
        auto b = std::optional<std::string>();

        auto result = drip::app::utils::and_both(a, b);

        CHECK_FALSE(result.has_value());
    }

    TEST_CASE("returns nullopt when both optionals are empty")
    {
        auto a = std::optional<int>();
        auto b = std::optional<std::string>();

        auto result = drip::app::utils::and_both(a, b);

        CHECK_FALSE(result.has_value());
    }

    TEST_CASE("moves values from optionals")
    {
        auto a = std::optional<std::string>("first");
        auto b = std::optional<std::string>("second");

        auto result = drip::app::utils::and_both(std::move(a), std::move(b));

        REQUIRE(result.has_value());
        CHECK(result->first == "first");
        CHECK(result->second == "second");
    }
}