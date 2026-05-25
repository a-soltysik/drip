add_library(drip_app STATIC)

target_sources(
        drip_app
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
)

target_link_libraries(
        drip_app
        PUBLIC
        drip::options
        drip::warnings
)

find_package(Vulkan REQUIRED)
target_link_system_libraries(
        drip_app
        PUBLIC
        drip::engine
        drip::common
        drip::sim
        glfw
        glm::glm
        imgui
        nlohmann_json
        nlohmann_json_schema_validator
        Vulkan::Vulkan
)

target_include_directories(drip_app PUBLIC ${CMAKE_CURRENT_LIST_DIR})
add_library(drip::app ALIAS drip_app)


set_target_properties(
        drip_app
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        FOLDER "Libraries")

