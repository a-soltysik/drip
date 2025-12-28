function(drip_setup_dependencies)

    CPMAddPackage("gh:fmtlib/fmt#12.1.0")
    CPMAddPackage("gh:g-truc/glm#1.0.2")
    CPMAddPackage("gh:glfw/glfw#3.4")
    CPMAddPackage(
            NAME
            ctre
            VERSION
            3.9.0
            GITHUB_REPOSITORY
            hanickadot/compile-time-regular-expressions)
    CPMAddPackage(
            NAME Boost
            VERSION 1.90.0
            URL https://github.com/boostorg/boost/releases/download/boost-1.90.0/boost-1.90.0-cmake.tar.xz
            URL_HASH SHA256=aca59f889f0f32028ad88ba6764582b63c916ce5f77b31289ad19421a96c555f
            OPTIONS
            "BOOST_INCLUDE_LIBRARIES exception\\\;stacktrace"
    )

    set(IMGUI_BUILD_GLFW_BINDING ON)
    set(IMGUI_BUILD_VULKAN_BINDING ON)

    CPMAddPackage("gh:ocornut/imgui@1.92.5")
    add_subdirectory(ext/imgui-cmake)

endfunction()
