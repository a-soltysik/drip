add_executable(drip)

target_sources(
        drip
        PRIVATE
        main.cpp)

target_link_system_libraries(
        drip
        PRIVATE
        fmt::fmt)

set_target_properties(
        drip
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


