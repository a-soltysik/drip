include(${CMAKE_SOURCE_DIR}/cmake/Cuda.cmake)

add_library(drip_sim STATIC)
add_library(drip::sim ALIAS drip_sim)

target_sources(
        drip_sim
        PRIVATE
        src/CudaExternalMemory.cu
        src/ExternalMemory.cu
        src/Simulation.cu
        src/Sph.cu
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

target_link_system_libraries(
        drip_sim
        PRIVATE
        drip::common
        glm::glm
)

drip_target_link_cuda(drip_sim)

target_compile_definitions(drip_sim
        PRIVATE
        GLM_ENABLE_EXPERIMENTAL
        $<$<PLATFORM_ID:Windows>:FMT_UNICODE=0>
)

set_target_properties(drip_sim PROPERTIES FOLDER "Libraries")
