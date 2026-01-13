#pragma once
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <glm/vec4.hpp>

#include "drip/simulation/Simulation.cuh"

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
        uint32_t count;
    };

    explicit Sph(SharedMemory sharedMemory, Domain domain);

    void update(float deltaTime) override;

private:
    struct InternalMemory
    {
        thrust::device_vector<glm::vec4> velocities;
        thrust::device_vector<glm::vec4> accelerations;
        thrust::device_vector<float> densities;
    };

    static auto createParticlePositions(Domain domain, size_t particleCount) -> thrust::host_vector<glm::vec4>;
    static auto createInternalMemory(const SharedMemory& sharedMemory) -> InternalMemory;
    static auto createFluidParticlesData(const SharedMemory& sharedMemory, InternalMemory& internalMemory)
        -> FluidParticlesData;
    static auto createSharedMemory(SharedMemory sharedMemory, Domain domain) -> SharedMemory;

    static constexpr auto threadsPerBlock = 256;

    [[nodiscard]] auto getBlocksPerGridForFluidParticles() const -> dim3;
    void computeExternalAccelerations() const;
    void computeDensities() const;
    void computePressureAccelerations() const;
    void computeViscosityAccelerations() const;
    void computeSurfaceTensionAccelerations() const;
    void updateVelocities(float deltaTime) const;
    void updatePositions(float deltaTime) const;
    void updateColors() const;

    SharedMemory _sharedMemory;
    InternalMemory _internalMemory;
    FluidParticlesData _data;
    Domain _domain;
};
}
