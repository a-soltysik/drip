#include <cuda_runtime.h>
#include <vector_types.h>

#include <cmath>
#include <glm/exponential.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/geometric.hpp>

#include "SimulationParameters.cuh"
#include "Sph.cuh"
#include "SphKernels.cuh"
#include "Utils.cuh"
#include "WendlandKernel.cuh"

namespace drip::sim
{
namespace kernel::constant
{
__constant__ Sph::FluidParticlesData particles;
}

void uploadFluidParticlesData(const Sph::FluidParticlesData& data)
{
    cudaMemcpyToSymbol(kernel::constant::particles, &data, sizeof(Sph::FluidParticlesData));
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

__global__ void computeDensities(SimulationParameters simulationParameters, NeighborGrid::DeviceView grid)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= simulationParameters.fluid.properties.particleCount)
    {
        return;
    }

    const auto position = constant::particles.positions[idx];

    auto density = 0.F;
    grid.forEachFluidNeighbor(
        position,
        constant::particles.positions,
        device::constant::wendlandRangeRatio * simulationParameters.fluid.properties.particle.smoothingRadius,
        [&density, &simulationParameters, position](const auto neighborId, const auto neighborPosition) {
            const auto offsetToNeighbour = neighborPosition - position;
            const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);

            if (distanceSquared > simulationParameters.fluid.properties.particle.neighborSearchRangeSquared)
            {
                return;
            }

            const auto distance = glm::sqrt(distanceSquared);
            const auto kernel =
                device::wendlandKernel(distance, simulationParameters.fluid.properties.particle.smoothingRadius);

            density += simulationParameters.fluid.properties.particle.mass * kernel;
        });

    constant::particles.densities[idx] = density;
}

__global__ void computePressureAccelerations(SimulationParameters simulationParameters, NeighborGrid::DeviceView grid)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= simulationParameters.fluid.properties.particleCount)
    {
        return;
    }
    const auto position = constant::particles.positions[idx];
    const auto density = constant::particles.densities[idx];
    const auto pressure = computeTaitPressure(density,
                                              simulationParameters.fluid.properties.density,
                                              simulationParameters.fluid.properties.speedOfSound);

    auto acceleration = glm::vec4 {};
    grid.forEachFluidNeighbor(
        position,
        constant::particles.positions,
        device::constant::wendlandRangeRatio * simulationParameters.fluid.properties.particle.smoothingRadius,
        [&acceleration, &simulationParameters, position, density, pressure](const auto neighborId,
                                                                            const auto neighborPosition) {
            const auto offsetToNeighbour = position - neighborPosition;
            const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);
            if (distanceSquared > simulationParameters.fluid.properties.particle.neighborSearchRangeSquared)
            {
                return;
            }

            const auto densityNeighbor = constant::particles.densities[neighborId];
            const auto distance = glm::sqrt(distanceSquared);
            const auto direction = distance > 0.F ? offsetToNeighbour / distance : glm::vec4(0.F, 1.F, 0.F, 0.F);
            const auto pressureNeighbor = computeTaitPressure(densityNeighbor,
                                                              simulationParameters.fluid.properties.density,
                                                              simulationParameters.fluid.properties.speedOfSound);

            const auto pressureTerm =
                (pressure / (density * density)) + (pressureNeighbor / (densityNeighbor * densityNeighbor));
            acceleration -=
                simulationParameters.fluid.properties.particle.mass * direction *
                device::wendlandDerivativeKernel(distance,
                                                 simulationParameters.fluid.properties.particle.smoothingRadius) *
                pressureTerm;
        });
    constant::particles.accelerations[idx] += acceleration;
}

__global__ void computeViscosityAccelerations(SimulationParameters simulationParameters, NeighborGrid::DeviceView grid)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= simulationParameters.fluid.properties.particleCount)
    {
        return;
    }
    const auto position = constant::particles.positions[idx];
    const auto velocity = constant::particles.velocities[idx];
    const auto density = constant::particles.densities[idx];

    auto acceleration = glm::vec4 {};
    grid.forEachFluidNeighbor(
        position,
        constant::particles.positions,
        device::constant::wendlandRangeRatio * simulationParameters.fluid.properties.particle.smoothingRadius,
        [&acceleration, &simulationParameters, position, velocity, density](const auto neighborId,
                                                                            const auto neighborPosition) {
            const auto offsetToNeighbour = position - neighborPosition;
            const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);
            if (distanceSquared > simulationParameters.fluid.properties.particle.neighborSearchRangeSquared)
            {
                return;
            }

            const auto neighborVelocity = constant::particles.velocities[neighborId];
            const auto velocityDifference = velocity - neighborVelocity;
            const auto compression = glm::dot(velocityDifference, offsetToNeighbour);
            if (compression >= 0.F)
            {
                return;
            }

            static constexpr auto epsilon = 0.01F;
            const auto distance = glm::sqrt(distanceSquared);
            const auto neighborDensity = constant::particles.densities[neighborId];
            const auto nu = 2.F * simulationParameters.fluid.properties.viscosity *
                            simulationParameters.fluid.properties.particle.smoothingRadius *
                            simulationParameters.fluid.properties.speedOfSound / (density + neighborDensity);

            const auto pi =
                -nu * compression /
                (distanceSquared + (epsilon * simulationParameters.fluid.properties.particle.smoothingRadius *
                                    simulationParameters.fluid.properties.particle.smoothingRadius));
            const auto direction = distance > 0.F ? offsetToNeighbour / distance : glm::vec4(0.F, 1.F, 0.F, 0.F);

            acceleration -=
                simulationParameters.fluid.properties.particle.mass * pi *
                device::wendlandDerivativeKernel(distance,
                                                 simulationParameters.fluid.properties.particle.smoothingRadius) *
                direction;
        });
    constant::particles.accelerations[idx] += acceleration;
}

__global__ void computeSurfaceTensionAccelerations(SimulationParameters simulationParameters,
                                                   NeighborGrid::DeviceView grid)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= simulationParameters.fluid.properties.particleCount)
    {
        return;
    }
    const auto position = constant::particles.positions[idx];

    auto acceleration = glm::vec4 {};
    grid.forEachFluidNeighbor(
        position,
        constant::particles.positions,
        device::constant::wendlandRangeRatio * simulationParameters.fluid.properties.particle.smoothingRadius,
        [&acceleration, &simulationParameters, position](const auto neighborId, const auto neighborPosition) {
            const auto offsetToNeighbour = position - neighborPosition;
            const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);
            if (distanceSquared > simulationParameters.fluid.properties.particle.neighborSearchRangeSquared)
            {
                return;
            }

            const auto distance = glm::sqrt(distanceSquared);
            const auto direction = distance > 0.F ? offsetToNeighbour / distance : glm::vec4(0.F, 1.F, 0.F, 0.F);

            acceleration +=
                simulationParameters.fluid.properties.particle.mass *
                device::wendlandKernel(distance, simulationParameters.fluid.properties.particle.smoothingRadius) *
                direction;
        });

    constant::particles.accelerations[idx] +=
        (simulationParameters.fluid.properties.surfaceTension / simulationParameters.fluid.properties.particle.mass) *
        acceleration;
}

__global__ void computeExternalAccelerations(SimulationParameters simulationParameters)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= simulationParameters.fluid.properties.particleCount)
    {
        return;
    }

    constant::particles.accelerations[idx] = glm::vec4 {simulationParameters.gravity, 0.F};
}

__global__ void updateVelocities(SimulationParameters simulationParameters, float dt)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= simulationParameters.fluid.properties.particleCount)
    {
        return;
    }
    constant::particles.velocities[idx] += constant::particles.accelerations[idx] * dt;

    const auto velocityMagnitude = glm::length(constant::particles.velocities[idx]);
    if (velocityMagnitude > simulationParameters.fluid.properties.maxVelocity)
    {
        constant::particles.velocities[idx] *= simulationParameters.fluid.properties.maxVelocity / velocityMagnitude;
    }
}

__global__ void updatePositions(SimulationParameters simulationParameters, float dt)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= simulationParameters.fluid.properties.particleCount)
    {
        return;
    }
    constant::particles.positions[idx] += constant::particles.velocities[idx] * dt;

    const auto position = glm::vec3 {constant::particles.positions[idx]};
    const auto clampedPosition =
        glm::clamp(position, simulationParameters.domain.bounds.min, simulationParameters.domain.bounds.max);
    constant::particles.positions[idx] = glm::vec4 {clampedPosition, constant::particles.positions[idx].w};
}

__global__ void updateColors(SimulationParameters simulationParameters)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= simulationParameters.fluid.properties.particleCount)
    {
        return;
    }

    const auto velocity = constant::particles.velocities[idx];
    const auto speed = glm::length(velocity);
    const auto normalizedSpeed = speed / simulationParameters.fluid.properties.maxVelocity;
    const auto color = utils::turboColormap(normalizedSpeed);
    constant::particles.colors[idx] = glm::vec4 {color, 0.0F};
}

}
}