#pragma once

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

    static auto create(SharedMemory sharedMemory) -> std::unique_ptr<Simulation>;

    Simulation() = default;
    Simulation(const Simulation&) = delete;
    Simulation(Simulation&&) = delete;
    auto operator=(const Simulation&) = delete;
    auto operator=(Simulation&&) = delete;
    virtual ~Simulation() noexcept = default;
    virtual void update(float deltaTime) = 0;
};

}
