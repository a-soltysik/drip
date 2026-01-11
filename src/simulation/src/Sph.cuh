#pragma once
#include <span>

#include "drip/simulation/Simulation.cuh"
#include "glm/vec4.hpp"

namespace drip::sim
{
class Sph : public Simulation
{
public:
    explicit Sph(SharedMemory sharedMemory);

    void update(float deltaTime) override;

private:
    struct FluidParticlesData
    {
        std::span<glm::vec4> positions;
        std::span<glm::vec4> colors;
        std::span<float> sizes;
    };

    SharedMemory _sharedMemory;
    FluidParticlesData _data;
};
}
