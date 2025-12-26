add_library(drip_common STATIC ${COMMON_SRC})
add_library(drip::common ALIAS drip_common)

target_sources(
        drip_common
        PRIVATE
        src/Logger.cpp
)

target_include_directories(
        drip_common
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)


find_package(Vulkan REQUIRED)
target_link_system_libraries(
        drip_common
        PUBLIC
        fmt::fmt)

target_link_libraries(
        drip_common
        PRIVATE
        drip::options
        drip::warnings
)

set_target_properties(drip_common PROPERTIES FOLDER "Libraries")
