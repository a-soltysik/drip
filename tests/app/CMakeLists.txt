add_executable(drip_app_tests)

target_sources(
        drip_app_tests
        PRIVATE
        main.cpp
        utils/IntervalCacheTest.cpp
        utils/FrameTimeManagerTest.cpp
        utils/UtilsTest.cpp
        config/SerializersTest.cpp
        config/JsonFileReaderTest.cpp
)

target_link_libraries(
        drip_app_tests
        PRIVATE
        drip::options
        drip::warnings
)

target_link_system_libraries(
        drip_app_tests
        PRIVATE
        drip::app
        doctest::doctest
)

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(drip_app_tests)
