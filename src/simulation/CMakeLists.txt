include(${CMAKE_SOURCE_DIR}/cmake/Cuda.cmake)

add_library(drip_sim SHARED)
add_library(drip::sim ALIAS drip_sim)

target_sources(
        drip_sim
        PRIVATE
        src/Simulation.cu
)

target_include_directories(drip_sim
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(
        drip_sim
        PRIVATE
        drip::options
        drip::warnings

)
drip_target_link_cuda(drip_sim)

target_compile_definitions(drip_sim
        PRIVATE
        DRIP_CUDA_EXPORTS
        GLM_ENABLE_EXPERIMENTAL
)

set_target_properties(
        drip_sim
        PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set_target_properties(drip_sim PROPERTIES FOLDER "Libraries")
