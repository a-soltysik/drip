#pragma once
#include <thrust/host_vector.h>

#include <glm/vec4.hpp>
#include <span>

#include "drip/simulation/Simulation.cuh"

namespace drip::sim
{
class Sph : public Simulation
{
public:
    explicit Sph(SharedMemory sharedMemory, Domain domain);

    void update(float deltaTime) override;

private:
    struct FluidParticlesData
    {
        std::span<glm::vec4> positions;
        std::span<glm::vec4> colors;
        std::span<float> sizes;
    };

    static auto createParticlePositions(Domain domain, size_t particleCount) -> thrust::host_vector<glm::vec4>;

    SharedMemory _sharedMemory;
    FluidParticlesData _data;
};
}
