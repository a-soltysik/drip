add_executable(drip main.cpp)
target_link_libraries(drip PRIVATE drip::app)

add_subdirectory(app)
add_subdirectory(common)
add_subdirectory(gfx)
add_subdirectory(simulation)

add_custom_target(copy_schema ALL
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/doc/simulation_config.schema.json
        $<TARGET_FILE_DIR:drip>/simulation_config.schema.json
        COMMENT "Copying JSON schema to output directory"
)
add_dependencies(drip copy_schema)