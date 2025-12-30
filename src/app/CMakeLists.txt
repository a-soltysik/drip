add_executable(drip)

target_sources(
        drip
        PRIVATE
        App.cpp
        main.cpp
        SystemSignalHandler.cpp
        Window.cpp
        mesh/InvertedCube.cpp)

target_link_libraries(
        drip
        PRIVATE
        drip::options
        drip::warnings
)

find_package(Vulkan REQUIRED)
target_link_system_libraries(
        drip
        PRIVATE
        drip::engine
        drip::common

        glfw
        glm
        imgui
        Vulkan::Vulkan)

set_target_properties(
        drip
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


