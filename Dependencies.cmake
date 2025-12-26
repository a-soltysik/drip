function(drip_setup_dependencies)

    CPMAddPackage("gh:fmtlib/fmt#12.1.0")
    CPMAddPackage("gh:g-truc/glm#1.0.2")

    set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
    CPMAddPackage("gh:glfw/glfw#3.4")
    CPMAddPackage(
            NAME
            ctre
            VERSION
            3.9.0
            GITHUB_REPOSITORY
            hanickadot/compile-time-regular-expressions)

    set(IMGUI_BUILD_GLFW_BINDING ON)
    set(IMGUI_BUILD_VULKAN_BINDING ON)

    CPMAddPackage("gh:ocornut/imgui@1.92.5")
    add_subdirectory(ext/imgui-cmake)

endfunction()
