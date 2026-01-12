add_executable(drip)

target_sources(
        drip
        PRIVATE
        input/KeyboardHandler.cpp
        input/MouseHandler.cpp

        mesh/InvertedCube.cpp

        ui/panel/StatisticsPanel.cpp
        ui/CameraHandler.cpp
        ui/MovementHandler.cpp
        ui/RotationHandler.cpp
        ui/Window.cpp

        utils/FrameTimeManager.cpp
        utils/Scene.cpp
        utils/SystemSignalHandler.cpp

        App.cpp
        main.cpp
)

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
        drip::sim

        glfw
        glm::glm
        imgui
        Vulkan::Vulkan)

target_include_directories(drip PRIVATE ${CMAKE_CURRENT_LIST_DIR})

set_target_properties(
        drip
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


