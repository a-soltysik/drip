add_library(drip_app STATIC)
add_library(drip::app ALIAS drip_app)

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

target_include_directories(
        drip_app
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}
        PRIVATE
        ${CMAKE_BINARY_DIR}/configured_files/include
)

target_link_system_libraries(
        drip_app
        PUBLIC
        drip::gfx
        drip::common
        drip::sim
        glfw
        glm::glm
        imgui
        nlohmann_json
        nlohmann_json_schema_validator
        Vulkan::Vulkan
)

drip_library(drip_app)

file(READ "${CMAKE_SOURCE_DIR}/doc/simulation_config.schema.json" JSON_SCHEMA)
string(REGEX REPLACE "[ \t\n\r]+" " " JSON_SCHEMA "${JSON_SCHEMA}")
string(STRIP "${JSON_SCHEMA}" JSON_SCHEMA)
configure_file(config/JsonSchema.hpp.in "${CMAKE_BINARY_DIR}/configured_files/include/drip/app/config/JsonSchema.hpp" @ONLY)