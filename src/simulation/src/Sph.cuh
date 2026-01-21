#pragma once
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector_types.h>

#include <glm/vec4.hpp>

#include "NeighborGrid.cuh"
#include "SimulationParameters.cuh"
#include "drip/simulation/Simulation.cuh"
#include "drip/simulation/SimulationConfig.cuh"

namespace drip::sim
{
class Sph : public Simulation
{
public:
    struct FluidParticlesData
    {
        glm::vec4* positions;
        glm::vec4* velocities;
        glm::vec4* accelerations;
        glm::vec4* colors;
        float* sizes;
        float* densities;
    };

    explicit Sph(SharedMemory sharedMemory, const SimulationConfig& parameters);

    void update(float deltaTime) override;

private:
    struct InternalMemory
    {
        thrust::device_vector<glm::vec4> velocities;
        thrust::device_vector<glm::vec4> accelerations;
        thrust::device_vector<float> densities;
    };

    static auto createParticlePositions(const SimulationConfig& parameters) -> thrust::host_vector<glm::vec4>;
    static auto createInternalMemory(const SharedMemory& sharedMemory) -> InternalMemory;
    static auto createFluidParticlesData(const SharedMemory& sharedMemory, InternalMemory& internalMemory)
        -> FluidParticlesData;
    static auto createSharedMemory(SharedMemory sharedMemory, const SimulationConfig& parameters) -> SharedMemory;
    static auto createSphParameters(const SimulationConfig& parameters) -> SimulationParameters;

    static constexpr auto threadsPerBlock = 256;

    [[nodiscard]] auto getBlocksPerGridForFluidParticles() const -> dim3;
    void computeExternalAccelerations() const;
    void computeDensities();
    void computePressureAccelerations();
    void computeViscosityAccelerations();
    void computeSurfaceTensionAccelerations();
    void updateVelocities(float deltaTime) const;
    void updatePositions(float deltaTime) const;
    void updateColors() const;
    void updateGrid();

    SharedMemory _sharedMemory;
    InternalMemory _internalMemory;
    FluidParticlesData _data;
    SimulationParameters _parameters;
    NeighborGrid _grid;
};
}
