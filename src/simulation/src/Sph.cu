#include <cuda_runtime_api.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <vector_types.h>

#include <cstddef>
#include <drip/common/log/LogMessageBuilder.hpp>
#include <drip/common/utils/format/GlmFormatter.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/ext/vector_uint3.hpp>
#include <glm/gtx/string_cast.hpp>

#include "NeighborKernels.cuh"
#include "SimulationParameters.cuh"
#include "Sph.cuh"
#include "SphKernels.cuh"
#include "WendlandKernel.cuh"
#include "drip/simulation/Simulation.cuh"
#include "drip/simulation/SimulationConfig.cuh"

namespace drip::sim
{

Sph::Sph(SharedMemory sharedMemory, const SimulationConfig& parameters)
    : _sharedMemory {createSharedMemory(std::move(sharedMemory), parameters)},
      _internalMemory {createInternalMemory(_sharedMemory)},
      _data {createFluidParticlesData(_sharedMemory, _internalMemory)},
      _parameters {createSphParameters(parameters)},
      _grid {_parameters.domain, 2.F * _parameters.fluid.properties.particle.smoothingRadius, _data.count}
{
    common::log::Info("Sph simulation created with {} particles and configuration:\n{}", _data.count, _parameters);
}

void Sph::update(float deltaTime)
{
    uploadSimulationParameters(_parameters);
    updateGrid();
    computeDensities();
    computeExternalAccelerations();
    computePressureAccelerations();
    computeViscosityAccelerations();
    computeSurfaceTensionAccelerations();

    updateVelocities(deltaTime);
    updatePositions(deltaTime);
    updateColors();

    cudaDeviceSynchronize();
}

auto Sph::createParticlePositions(const SimulationConfig& parameters) -> thrust::host_vector<glm::vec4>
{
    const auto fluidScale = parameters.fluid.bounds.max - parameters.fluid.bounds.min;
    const auto particleCount = glm::uvec3 {glm::floor(fluidScale / parameters.fluid.properties.spacing)};
    common::log::Info("Particle grid: {}", particleCount);

    auto hostPositions = thrust::host_vector<glm::vec4>(particleCount.x * particleCount.y * particleCount.z);

    for (auto z = size_t {}; z < particleCount.z; ++z)
    {
        for (auto y = size_t {}; y < particleCount.y; ++y)
        {
            for (auto x = size_t {}; x < particleCount.x; ++x)
            {
                const auto idx = x + y * particleCount.x + z * particleCount.x * particleCount.y;
                hostPositions[idx] = glm::vec4 {
                    parameters.fluid.bounds.min + (glm::vec3 {x, y, z} + 0.5F) * parameters.fluid.properties.spacing,
                    1.F};
            }
        }
    }
    return hostPositions;
}

auto Sph::createInternalMemory(const SharedMemory& sharedMemory) -> InternalMemory
{
    const auto particlesCount = sharedMemory.positions->getSize() / sizeof(glm::vec4);
    common::log::Info("Initializing internal buffers for {} particles", particlesCount);
    return {.velocities = thrust::device_vector<glm::vec4>(particlesCount, glm::vec4 {}),
            .accelerations = thrust::device_vector<glm::vec4>(particlesCount, glm::vec4 {}),
            .densities = thrust::device_vector<float>(particlesCount, float {})};
}

auto Sph::createFluidParticlesData(const SharedMemory& sharedMemory, InternalMemory& internalMemory)
    -> FluidParticlesData
{
    return {.positions = static_cast<glm::vec4*>(sharedMemory.positions->getData()),
            .velocities = thrust::raw_pointer_cast(internalMemory.velocities.data()),
            .accelerations = thrust::raw_pointer_cast(internalMemory.accelerations.data()),
            .colors = static_cast<glm::vec4*>(sharedMemory.colors->getData()),
            .sizes = static_cast<float*>(sharedMemory.sizes->getData()),
            .densities = thrust::raw_pointer_cast(internalMemory.densities.data()),
            .count = static_cast<uint32_t>(internalMemory.velocities.size())};
}

auto Sph::createSharedMemory(SharedMemory sharedMemory, const SimulationConfig& parameters) -> SharedMemory
{
    const auto particlesCount = sharedMemory.positions->getSize() / sizeof(glm::vec4);
    const auto hostPositions = createParticlePositions(parameters);
    thrust::copy(hostPositions.begin(),
                 hostPositions.end(),
                 thrust::device_ptr<glm::vec4>(static_cast<glm::vec4*>(sharedMemory.positions->getData())));
    thrust::fill_n(thrust::device,
                   static_cast<glm::vec4*>(sharedMemory.colors->getData()),
                   particlesCount,
                   glm::vec4 {});
    thrust::fill_n(thrust::device,
                   static_cast<float*>(sharedMemory.sizes->getData()),
                   particlesCount,
                   parameters.fluid.properties.spacing / 2);
    return sharedMemory;
}

auto Sph::createSphParameters(const SimulationConfig& parameters) -> SimulationParameters
{
    const auto fluidScale = parameters.fluid.bounds.max - parameters.fluid.bounds.min;
    const auto particleGrid = glm::uvec3 {glm::floor(fluidScale / parameters.fluid.properties.spacing)};
    const auto particleCount = particleGrid.x * particleGrid.y * particleGrid.z;
    const auto volume = fluidScale.x * fluidScale.y * fluidScale.z;
    const auto particleMass = volume * parameters.fluid.properties.density / static_cast<float>(particleCount);

    return {
        .domain = {.bounds =
                       {
                           .min = parameters.domain.bounds.min,
                           .max = parameters.domain.bounds.max,
                       }},
        .fluid = {.bounds =
                      {
                          .min = parameters.fluid.bounds.min,
                          .max = parameters.fluid.bounds.max,
                      }, .properties =
                      {
                          .particle = {.mass = particleMass,
                                       .radius = parameters.fluid.properties.spacing / 2,
                                       .smoothingRadius = parameters.fluid.properties.smoothingRadius,
                                       .neighborSearchRangeSquared = device::constant::wendlandRangeRatio *
                                                                     device::constant::wendlandRangeRatio *
                                                                     parameters.fluid.properties.smoothingRadius *
                                                                     parameters.fluid.properties.smoothingRadius},
                          .density = parameters.fluid.properties.density,
                          .surfaceTension = parameters.fluid.properties.surfaceTension,
                          .viscosity = parameters.fluid.properties.viscosity,
                          .maxVelocity = parameters.fluid.properties.maxVelocity,
                          .speedOfSound = parameters.fluid.properties.speedOfSound,
                      }},
        .gravity = parameters.gravity
    };
}

auto Sph::getBlocksPerGridForFluidParticles() const -> dim3
{
    return {(_data.count + threadsPerBlock - 1) / threadsPerBlock};
}

void Sph::computeExternalAccelerations() const
{
    kernel::computeExternalAccelerations<<<getBlocksPerGridForFluidParticles(), threadsPerBlock>>>(_data);
}

void Sph::computeDensities()
{
    kernel::computeDensities<<<getBlocksPerGridForFluidParticles(), threadsPerBlock>>>(_data, _grid.toDeviceView());
}

void Sph::computePressureAccelerations()
{
    kernel::computePressureAccelerations<<<getBlocksPerGridForFluidParticles(), threadsPerBlock>>>(
        _data,
        _grid.toDeviceView());
}

void Sph::computeViscosityAccelerations()
{
    kernel::computeViscosityAccelerations<<<getBlocksPerGridForFluidParticles(), threadsPerBlock>>>(
        _data,
        _grid.toDeviceView());
}

void Sph::computeSurfaceTensionAccelerations()
{
    kernel::computeSurfaceTensionAccelerations<<<getBlocksPerGridForFluidParticles(), threadsPerBlock>>>(
        _data,
        _grid.toDeviceView());
}

void Sph::updateVelocities(float deltaTime) const
{
    kernel::updateVelocities<<<getBlocksPerGridForFluidParticles(), threadsPerBlock>>>(_data, deltaTime);
}

void Sph::updatePositions(float deltaTime) const
{
    kernel::updatePositions<<<getBlocksPerGridForFluidParticles(), threadsPerBlock>>>(_data, deltaTime);
}

void Sph::updateColors() const
{
    kernel::updateColors<<<getBlocksPerGridForFluidParticles(), threadsPerBlock>>>(_data);
}

void Sph::updateGrid()
{
    _grid.update({getBlocksPerGridForFluidParticles(), threadsPerBlock}, _data.positions, _data.count);
}
}
