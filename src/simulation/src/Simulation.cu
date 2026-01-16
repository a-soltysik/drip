#include "Sph.cuh"
#include "drip/simulation/Simulation.cuh"
#include "drip/simulation/SimulationConfig.cuh"

namespace drip::sim
{

auto Simulation::create(SharedMemory sharedMemory, const SimulationConfig& parameters) -> std::unique_ptr<Simulation>
{
    auto result = std::make_unique<Sph>(std::move(sharedMemory), parameters);
    return result;
}

}
