# drip

[![Formatting Check](https://github.com/a-soltysik/drip/actions/workflows/formatting.yml/badge.svg)](https://github.com/a-soltysik/drip/actions/workflows/formatting.yml)
[![Static Analysis](https://github.com/a-soltysik/drip/actions/workflows/static-analysis.yml/badge.svg)](https://github.com/a-soltysik/drip/actions/workflows/static-analysis.yml)
[![Build](https://github.com/a-soltysik/drip/actions/workflows/build.yml/badge.svg)](https://github.com/a-soltysik/drip/actions/workflows/build.yml)
[![Test](https://github.com/a-soltysik/drip/actions/workflows/test.yml/badge.svg)](https://github.com/a-soltysik/drip/actions/workflows/test.yml)

Real-time 3D fluid simulation using **Smoothed Particle Hydrodynamics (SPH)** on the GPU, rendered with Vulkan.

The simulation runs entirely on CUDA — particle interactions, pressure, viscosity, and surface tension are computed each
frame. Rendering uses a custom Vulkan renderer with an ImGui overlay for live statistics.

## Features

- GPU-accelerated SPH simulation (CUDA)
- Real-time Vulkan rendering
- CFL-adaptive timestep — the simulation automatically adjusts the timestep to maintain stability
- Configurable fluid properties: density, viscosity, surface tension, speed of sound
- Configurable domain and initial fluid volume
- Live statistics panel (frame time, frame rate)
- JSON configuration with schema validation

## Building

The project uses CMake presets. Choose the preset that matches your platform and desired configuration.

**Windows (MSVC):**

```sh
cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-release
```

**Linux (GCC):**

```sh
cmake --preset linux-gcc-release
cmake --build --preset linux-gcc-release
```

**Linux (Clang):**

```sh
cmake --preset linux-clang-release
cmake --build --preset linux-clang-release
```

The compiled binary is placed in `build/<preset>/src/`.

### Available presets

```sh
cmake --list-presets
```

## Usage

```sh
# Run with default simulation parameters
drip

# Run with a custom configuration file
drip path/to/config.json
```

### Controls

| Input                    | Action              |
|--------------------------|---------------------|
| Left mouse button + drag | Rotate camera       |
| W / S                    | Move forward / back |
| A / D                    | Move left / right   |
| Space / Left Shift       | Move up / down      |

## Configuration

Simulation parameters can be provided via a JSON file. All fields are optional — omitted values fall back to defaults.

```json
{
  "domain": {
    "bounds": {
      "min": {
        "x": -1.0,
        "y": -1.0,
        "z": -1.0
      },
      "max": {
        "x": 1.0,
        "y": 1.0,
        "z": 1.0
      }
    }
  },
  "fluid": {
    "bounds": {
      "min": {
        "x": -0.5,
        "y": -0.5,
        "z": -0.5
      },
      "max": {
        "x": 0.5,
        "y": 0.5,
        "z": 0.5
      }
    },
    "properties": {
      "spacing": 0.08,
      "smoothingRadius": 0.08,
      "density": 1000.0,
      "viscosity": 0.1,
      "surfaceTension": 1.0,
      "speedOfSound": 50.0,
      "maxVelocity": 10.0
    }
  },
  "environment": {
    "gravity": {
      "x": 0.0,
      "y": -9.81,
      "z": 0.0
    }
  }
}
```

### Parameters

| Parameter                          | Description                                                                            | Default         |
|------------------------------------|----------------------------------------------------------------------------------------|-----------------|
| `domain.bounds`                    | The bounding box that confines the fluid                                               | `[-1, 1]³`      |
| `fluid.bounds`                     | Initial volume where particles are spawned                                             | `[-1, 1]³`      |
| `fluid.properties.spacing`         | Distance between particles at initialization; also determines rendered particle radius | `0.08`          |
| `fluid.properties.smoothingRadius` | SPH smoothing length — controls how far particles influence each other                 | `0.08`          |
| `fluid.properties.density`         | Rest density of the fluid (kg/m³)                                                      | `1000.0`        |
| `fluid.properties.viscosity`       | Viscosity coefficient — higher values produce thicker fluid                            | `0.1`           |
| `fluid.properties.surfaceTension`  | Surface tension coefficient                                                            | `1.0`           |
| `fluid.properties.speedOfSound`    | Speed of sound in the fluid — affects pressure stiffness                               | `50.0`          |
| `fluid.properties.maxVelocity`     | Velocity clamp — prevents instability at large timesteps                               | `10.0`          |
| `environment.gravity`              | Gravitational acceleration vector                                                      | `(0, -9.81, 0)` |

The JSON schema is available at [`doc/simulation_config.schema.json`](doc/simulation_config.schema.json).

## Running tests

```sh
cmake --preset linux-gcc-debug
cmake --build --preset linux-gcc-debug
ctest --preset linux-gcc-debug
```

## Tech stack

- **CUDA** — GPU kernels for SPH neighbor search and force integration
- **Vulkan** — low-level graphics API via `vulkan.hpp`
- **GLFW** — window and input management
- **GLM** — mathematics
- **ImGui** — immediate-mode GUI overlay
- **nlohmann/json** + **json-schema-validator** — typed config parsing with validation
- **Boost** — stacktrace and exception utilities
- **fmt** — formatting
- **doctest** — unit testing
