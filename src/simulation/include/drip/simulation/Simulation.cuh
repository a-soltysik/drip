#pragma once

#include <glm/vec3.hpp>
#include <memory>

#include "ExternalMemory.cuh"

namespace drip::sim
{

class Simulation
{
public:
    struct SharedMemory
    {
        std::unique_ptr<ExternalMemory> positions;
        std::unique_ptr<ExternalMemory> colors;
        std::unique_ptr<ExternalMemory> sizes;
    };

    struct Domain
    {
        glm::vec3 min;
        glm::vec3 max;
        glm::uvec3 sampling;
    };

    static auto create(SharedMemory sharedMemory, Domain domain) -> std::unique_ptr<Simulation>;

    Simulation() = default;
    Simulation(const Simulation&) = delete;
    Simulation(Simulation&&) = delete;
    auto operator=(const Simulation&) = delete;
    auto operator=(Simulation&&) = delete;
    virtual ~Simulation() noexcept = default;
    virtual void update(float deltaTime) = 0;
};

}
