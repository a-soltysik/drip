# CLAUDE.md - Project Context for AI Code Review

## Project Overview

**drip** is a real-time fluid simulation application implementing the WCSPH (Weakly Compressible Smoothed Particle Hydrodynamics) algorithm with GPU acceleration via CUDA and real-time visualization using Vulkan.

- **Languages**: C++23 (host code), CUDA C++20 (GPU kernels)
- **Build System**: CMake 3.25+ with CMakePresets and CPM for dependency management
- **Platforms**: Windows (MSVC), Linux (GCC, Clang)

## Repository Structure

```
drip/
├── src/
│   ├── app/           # Main application executable
│   ├── common/        # Shared utilities library (drip::common)
│   ├── gfx/           # Vulkan graphics library (drip::gfx)
│   └── simulation/    # CUDA SPH simulation library (drip::sim)
├── cmake/             # CMake modules and configuration
├── ext/               # External dependencies (imgui-cmake)
├── .github/           # GitHub Actions workflows
├── CMakeLists.txt     # Root CMake configuration
├── CMakePresets.json  # Build presets for different configurations
├── Dependencies.cmake # CPM dependency declarations
└── ProjectOptions.cmake # Project-specific CMake options
```

## Module Details

### `src/simulation/` - SPH Simulation (drip::sim)

CUDA-based fluid simulation implementing WCSPH algorithm.

**Key Files:**
- `Sph.cu/cuh` - Main SPH simulation class managing particle data and simulation loop
- `SphKernels.cu/cuh` - CUDA kernels for density, pressure, viscosity, surface tension calculations
- `NeighborGrid.cu/cuh` - Spatial hashing grid for O(n) neighbor search
- `WendlandKernel.cuh` - Wendland C2 smoothing kernel implementation
- `SimulationParameters.cuh` - Internal simulation parameters with fmt formatters
- `include/drip/simulation/` - Public API headers

**Algorithm Components:**
- Tait equation of state for pressure calculation
- Neighbor grid with cell-based spatial hashing
- Wendland C2 kernel (compact support, smooth derivatives)
- Artificial viscosity model
- Surface tension via cohesion forces

**CUDA Configuration:**
- 256 threads per block
- Thrust library for device vectors and sorting
- `__constant__` memory for simulation parameters
- External memory support for Vulkan interop

### `src/gfx/` - Vulkan Graphics (drip::gfx)

Vulkan-based rendering engine with ImGui integration.

**Key Components:**
- `vulkan/core/Context.hpp` - Vulkan instance, device, and resource management
- `vulkan/core/Device.hpp` - Physical/logical device wrapper
- `vulkan/core/SwapChain.hpp` - Swapchain management
- `vulkan/pipeline/` - Pipeline, shader, command buffer handling
- `vulkan/memory/` - Buffer, descriptor, and shared buffer management
- `resource/` - Mesh, texture, renderable abstractions
- `scene/` - Camera, lights, scene graph
- `gui/` - ImGui panel system
- `rendering/` - Render systems (mesh, particles, GUI)

**Key Features:**
- Double-buffered rendering (maxFramesInFlight = 2)
- Shared buffer support for CUDA-Vulkan interop (via external memory)
- Point-based particle rendering
- Turbo colormap for velocity visualization

### `src/app/` - Application

Main executable integrating simulation and rendering.

**Key Files:**
- `main.cpp` - Entry point, exception handling
- `App.hpp/cpp` - Application lifecycle, main loop
- `ui/Window.hpp/cpp` - GLFW window wrapper
- `ui/CameraHandler.hpp/cpp` - Camera controls
- `config/JsonFileReader.hpp` - JSON configuration loading
- `config/Serializers.hpp` - nlohmann/json serializers for SimulationConfig
- `utils/Scene.hpp/cpp` - Scene setup helper

**Configuration:**
- Accepts optional JSON config file as command line argument
- Default parameters defined in `SimulationConfig.cuh`

### `src/common/` - Common Utilities (drip::common)

Shared utilities across all modules.

**Components:**
- `log/Logger.hpp` - Thread-safe async logger with multiple sinks
- `log/LogMessageBuilder.hpp` - Fluent log message builder
- `log/sink/` - Console and file log sinks
- `utils/Signal.hpp` - Type-safe signal/slot system
- `utils/Timer.hpp` - High-resolution timer
- `utils/Assert.hpp` - Custom assertion macros with stacktrace support
- `utils/format/` - fmt formatters (GLM, exceptions, stacktrace)

### CMake Options

| Option | Description |
|--------|-------------|
| `DRIP_ENABLE_WARNINGS` | Enable compiler warnings |
| `DRIP_ENABLE_WARNINGS_AS_ERRORS` | Treat warnings as errors |
| `DRIP_ENABLE_CLANG_TIDY` | Enable clang-tidy |
| `DRIP_ENABLE_CPPCHECK` | Enable cppcheck |
| `DRIP_ENABLE_SANITIZER_*` | Enable sanitizers (ADDRESS, UNDEFINED, LEAK, THREAD, MEMORY) |
| `DRIP_ENABLE_IPO` | Enable interprocedural optimization |
| `DRIP_ENABLE_CACHE` | Enable ccache |
| `DRIP_CUDA_ENABLE_DEBUG` | Enable CUDA debug symbols (-G) |
| `DRIP_CUDA_ENABLE_LINEINFO` | Enable CUDA line info |

## Dependencies

Managed via CPM (CMake Package Manager):

| Library | Version | Purpose |
|---------|---------|---------|
| fmt | 12.1.0 | String formatting |
| glm | 1.0.2 | Math library (vectors, matrices) |
| glfw | 3.4 | Window/input management |
| nlohmann/json | 3.12.0 | JSON parsing |
| Boost | 1.90.0 | exception, stacktrace, thread |
| ImGui | 1.92.5 | GUI framework |

**System Dependencies:**
- CUDA Toolkit (for simulation)
- Vulkan SDK (for graphics)
- Ninja (recommended build tool)

## Code Style

### Formatting
- **Tool**: clang-format
- **Column limit**: 120
- **Indent**: 4 spaces
- **Braces**: Allman style (always on new line)
- **Trailing return types**: Preferred (`auto foo() -> int`)

### Naming Conventions
- Classes: `PascalCase`
- Functions/Methods: `camelCase`
- Variables: `camelCase`
- Private members: `_prefixedWithUnderscore`
- Constants: `camelCase`
- Macros: `UPPER_SNAKE_CASE`
- Namespaces: `lowercase`

### Static Analysis
- clang-tidy with extensive checks (see `.clang-tidy`)
- cppcheck

## Key Patterns

### Signal/Slot System
```cpp
// Declaration
common::signal::Signal<> mainLoopIterationStarted;

// Connection
auto receiver = signal.connect([](){ /* handler */ });

// Emission
signal.registerSender()();
```

### Logging
```cpp
common::log::Info("Message with {} format", value);
common::log::Error("Error!").withCurrentException().withStacktraceFromCurrentException();
```

### CUDA Kernels
- Parameters via `__constant__` memory
- Standard pattern: check bounds, compute per-particle
- Neighbor iteration via `NeighborGrid::DeviceView::forEachFluidNeighbor()`

### Vulkan-CUDA Interop
- `ParticlesRenderable` creates shared Vulkan buffers
- `ExternalMemory` wraps platform-specific handles (HANDLE on Windows, fd on Linux)
- Simulation writes directly to shared memory, no CPU round-trip

## CI/CD

GitHub Actions workflows:
- `build.yml` - Build on Linux with GCC
- `formatting.yml` - Check clang-format
- `static-analysis.yml` - Run clang-tidy and cppcheck
- `claude.yml` / `claude-code-review.yml` - AI-assisted code review

## Architecture Notes

### Data Flow
1. `App::run()` initializes window, Vulkan context, scene, simulation
2. `App::mainLoop()` runs the render loop:
   - Process input
   - Update camera
   - `Simulation::update(dt)` - CUDA kernels update particle positions/velocities
   - `Context::makeFrame()` - Vulkan renders the scene

### Memory Architecture
- Particle positions/colors/sizes live in Vulkan buffers
- CUDA accesses these via external memory API
- No explicit synchronization needed (single-threaded render loop)
