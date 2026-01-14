#include <cuda_runtime.h>
#include <vector_types.h>

#include <cmath>
#include <cstdint>
#include <glm/exponential.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/geometric.hpp>

#include "SimulationParameters.cuh"
#include "Sph.cuh"
#include "SphKernels.cuh"
#include "WendlandKernel.cuh"
#include "drip/simulation/Simulation.cuh"

namespace drip::sim
{
namespace kernel::constant
{
__constant__ SimulationParameters simulationParameters;
}

void uploadSimulationParameters(const SimulationParameters& parameters)
{
    cudaMemcpyToSymbol(kernel::constant::simulationParameters, &parameters, sizeof(SimulationParameters));
}

namespace kernel
{
__device__ auto computeTaitPressure(float density, float restDensity, float speedOfSound) -> float
{
    static constexpr auto gamma = 7.F;
    const auto B = restDensity * speedOfSound * speedOfSound / gamma;
    const auto densityRatio = density / restDensity;
    return B * (powf(densityRatio, gamma) - 1.F);
}

__global__ void computeDensities(Sph::FluidParticlesData particles)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.count)
    {
        return;
    }

    const auto position = particles.positions[idx];

    auto density = 0.F;
    for (auto i = uint32_t {}; i < particles.count; i++)
    {
        const auto neighborPosition = particles.positions[i];
        const auto offsetToNeighbour = neighborPosition - position;
        const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);

        if (distanceSquared > constant::simulationParameters.fluid.properties.particle.neighborSearchRangeSquared)
        {
            continue;
        }

        const auto distance = glm::sqrt(distanceSquared);
        const auto kernel =
            device::wendlandKernel(distance, constant::simulationParameters.fluid.properties.particle.smoothingRadius);

        density += constant::simulationParameters.fluid.properties.particle.mass * kernel;
    }
    particles.densities[idx] = density;
}

__global__ void computePressureAccelerations(Sph::FluidParticlesData particles)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.count)
    {
        return;
    }
    const auto position = particles.positions[idx];
    const auto density = particles.densities[idx];
    const auto pressure = computeTaitPressure(density,
                                              constant::simulationParameters.fluid.properties.density,
                                              constant::simulationParameters.fluid.properties.speedOfSound);

    auto acceleration = glm::vec4 {};
    for (auto i = uint32_t {}; i < particles.count; i++)
    {
        const auto neighborPosition = particles.positions[i];
        const auto offsetToNeighbour = position - neighborPosition;
        const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);
        if (distanceSquared > constant::simulationParameters.fluid.properties.particle.neighborSearchRangeSquared)
        {
            continue;
        }

        const auto densityNeighbor = particles.densities[i];
        const auto distance = glm::sqrt(distanceSquared);
        const auto direction = distance > 0.F ? offsetToNeighbour / distance : glm::vec4(0.F, 1.F, 0.F, 0.F);
        const auto pressureNeighbor = computeTaitPressure(densityNeighbor,
                                                          constant::simulationParameters.fluid.properties.density,
                                                          constant::simulationParameters.fluid.properties.speedOfSound);

        const auto pressureTerm =
            (pressure / (density * density)) + (pressureNeighbor / (densityNeighbor * densityNeighbor));
        acceleration -=
            constant::simulationParameters.fluid.properties.particle.mass * direction *
            device::wendlandDerivativeKernel(distance,
                                             constant::simulationParameters.fluid.properties.particle.smoothingRadius) *
            pressureTerm;
    }
    particles.accelerations[idx] += acceleration;
}

__global__ void computeViscosityAccelerations(Sph::FluidParticlesData particles)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.count)
    {
        return;
    }
    const auto position = particles.positions[idx];
    const auto velocity = particles.velocities[idx];
    const auto density = particles.densities[idx];

    auto acceleration = glm::vec4 {};
    for (auto i = uint32_t {}; i < particles.count; i++)
    {
        const auto neighborPosition = particles.positions[i];
        const auto offsetToNeighbour = position - neighborPosition;
        const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);
        if (distanceSquared > constant::simulationParameters.fluid.properties.particle.neighborSearchRangeSquared)
        {
            continue;
        }

        const auto neighborVelocity = particles.velocities[i];
        const auto velocityDifference = velocity - neighborVelocity;
        const auto compression = glm::dot(velocityDifference, offsetToNeighbour);
        if (compression >= 0.F)
        {
            continue;
        }

        static constexpr auto epsilon = 0.01F;
        const auto distance = glm::sqrt(distanceSquared);
        const auto neighborDensity = particles.densities[i];
        const auto nu = 2.F * constant::simulationParameters.fluid.properties.viscosity *
                        constant::simulationParameters.fluid.properties.particle.smoothingRadius *
                        constant::simulationParameters.fluid.properties.speedOfSound / (density + neighborDensity);

        const auto pi =
            -nu * compression /
            (distanceSquared + (epsilon * constant::simulationParameters.fluid.properties.particle.smoothingRadius *
                                constant::simulationParameters.fluid.properties.particle.smoothingRadius));
        const auto direction = distance > 0.F ? offsetToNeighbour / distance : glm::vec4(0.F, 1.F, 0.F, 0.F);

        acceleration -=
            constant::simulationParameters.fluid.properties.particle.mass * pi *
            device::wendlandDerivativeKernel(distance,
                                             constant::simulationParameters.fluid.properties.particle.smoothingRadius) *
            direction;
    }
    particles.accelerations[idx] += acceleration;
}

__global__ void computeSurfaceTensionAccelerations(Sph::FluidParticlesData particles)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.count)
    {
        return;
    }
    const auto position = particles.positions[idx];

    auto acceleration = glm::vec4 {};
    for (auto i = uint32_t {}; i < particles.count; i++)
    {
        const auto neighborPosition = particles.positions[i];
        const auto offsetToNeighbour = position - neighborPosition;
        const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);
        if (distanceSquared > constant::simulationParameters.fluid.properties.particle.neighborSearchRangeSquared)
        {
            continue;
        }

        const auto distance = glm::sqrt(distanceSquared);
        const auto direction = distance > 0.F ? offsetToNeighbour / distance : glm::vec4(0.F, 1.F, 0.F, 0.F);

        acceleration +=
            constant::simulationParameters.fluid.properties.particle.mass *
            device::wendlandKernel(distance, constant::simulationParameters.fluid.properties.particle.smoothingRadius) *
            direction;
    }
    particles.accelerations[idx] += (constant::simulationParameters.fluid.properties.surfaceTension /
                                     constant::simulationParameters.fluid.properties.particle.mass) *
                                    acceleration;
}

__global__ void computeExternalAccelerations(Sph::FluidParticlesData particles)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.count)
    {
        return;
    }

    particles.accelerations[idx] = glm::vec4 {constant::simulationParameters.gravity, 0.F};
}

__global__ void updateVelocities(Sph::FluidParticlesData particles, float dt)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particles.count)
    {
        return;
    }
    particles.velocities[idx] += particles.accelerations[idx] * dt;

    const auto velocityMagnitude = glm::length(particles.velocities[idx]);
    if (velocityMagnitude > constant::simulationParameters.fluid.properties.maxVelocity)
    {
        particles.velocities[idx] *= constant::simulationParameters.fluid.properties.maxVelocity / velocityMagnitude;
    }
}

__global__ void updatePositions(Sph::FluidParticlesData particles, float dt)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particles.count)
    {
        return;
    }
    particles.positions[idx] += particles.velocities[idx] * dt;

    const auto position = glm::vec3 {particles.positions[idx]};
    const auto clampedPosition = glm::clamp(position,
                                            constant::simulationParameters.domain.bounds.min,
                                            constant::simulationParameters.domain.bounds.max);
    particles.positions[idx] = glm::vec4 {clampedPosition, particles.positions[idx].w};
}

__global__ void updateColors(Sph::FluidParticlesData particles)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particles.count)
    {
        return;
    }

    static constexpr auto violet = float3 {.x = 0.5F, .y = 0.F, .z = 1.F};
    particles.colors[idx] = glm::vec4 {violet.x, violet.y, violet.z, 0.F};
}

}
}