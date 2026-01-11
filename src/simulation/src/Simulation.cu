#include "Sph.cuh"
#include "drip/simulation/Simulation.cuh"

namespace drip::sim
{

auto Simulation::create(SharedMemory sharedMemory) -> std::unique_ptr<Simulation>
{
    auto result = std::make_unique<Sph>(std::move(sharedMemory));
    return result;
}

}
