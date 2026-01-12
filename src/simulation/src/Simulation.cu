#include "Sph.cuh"
#include "drip/simulation/Simulation.cuh"

namespace drip::sim
{

auto Simulation::create(SharedMemory sharedMemory, Domain domain) -> std::unique_ptr<Simulation>
{
    auto result = std::make_unique<Sph>(std::move(sharedMemory), domain);
    return result;
}

}
