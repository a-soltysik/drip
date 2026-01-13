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
#include <glm/gtx/string_cast.hpp>

#include "Sph.cuh"
#include "SphKernels.cuh"

namespace drip::sim
{

Sph::Sph(SharedMemory sharedMemory, Domain domain)
    : _sharedMemory {createSharedMemory(std::move(sharedMemory), domain)},
      _internalMemory {createInternalMemory(_sharedMemory)},
      _data {createFluidParticlesData(_sharedMemory, _internalMemory)},
      _domain {domain}
{
    common::log::Info("Sph simulation created with {} particles", _data.count);
}

void Sph::update(float deltaTime)
{
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

auto Sph::createParticlePositions(Domain domain, size_t particleCount) -> thrust::host_vector<glm::vec4>
{
    common::log::Info("Particle grid: {}", domain.sampling);

    const auto domainSize = domain.max - domain.min;
    const auto spacing = domainSize / glm::vec3 {domain.sampling};

    auto hostPositions = thrust::host_vector<glm::vec4>(particleCount);

    for (auto z = size_t {}; z < domain.sampling.z; ++z)
    {
        for (auto y = size_t {}; y < domain.sampling.y; ++y)
        {
            for (auto x = size_t {}; x < domain.sampling.x; ++x)
            {
                const auto idx = x + y * domain.sampling.x + z * domain.sampling.x * domain.sampling.y;
                hostPositions[idx] = glm::vec4 {domain.min + (glm::vec3 {x, y, z} + 0.5F) * spacing, 1.F};
            }
        }
    }
    return hostPositions;
}

auto Sph::createInternalMemory(const SharedMemory& sharedMemory) -> InternalMemory
{
    const auto particlesCount = sharedMemory.positions->getSize() / sizeof(glm::vec4);
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

auto Sph::createSharedMemory(SharedMemory sharedMemory, Domain domain) -> SharedMemory
{
    const auto particlesCount = sharedMemory.positions->getSize() / sizeof(glm::vec4);
    const auto hostPositions = createParticlePositions(domain, particlesCount);
    static constexpr auto radius = 0.05F;
    thrust::copy(hostPositions.begin(),
                 hostPositions.end(),
                 thrust::device_ptr<glm::vec4>(static_cast<glm::vec4*>(sharedMemory.positions->getData())));
    thrust::fill_n(thrust::device,
                   static_cast<glm::vec4*>(sharedMemory.colors->getData()),
                   particlesCount,
                   glm::vec4 {});
    thrust::fill_n(thrust::device, static_cast<float*>(sharedMemory.sizes->getData()), particlesCount, radius);
    return sharedMemory;
}

auto Sph::getBlocksPerGridForFluidParticles() const -> dim3
{
    return {(_data.count + threadsPerBlock - 1) / threadsPerBlock};
}

void Sph::computeExternalAccelerations() const
{
    kernel::computeExternalAccelerations<<<getBlocksPerGridForFluidParticles(), threadsPerBlock>>>(_data);
}

void Sph::computeDensities() const
{
    kernel::computeDensities<<<getBlocksPerGridForFluidParticles(), threadsPerBlock>>>(_data);
}

void Sph::computePressureAccelerations() const
{
    kernel::computePressureAccelerations<<<getBlocksPerGridForFluidParticles(), threadsPerBlock>>>(_data);
}

void Sph::computeViscosityAccelerations() const
{
    kernel::computeViscosityAccelerations<<<getBlocksPerGridForFluidParticles(), threadsPerBlock>>>(_data);
}

void Sph::computeSurfaceTensionAccelerations() const
{
    kernel::computeSurfaceTensionAccelerations<<<getBlocksPerGridForFluidParticles(), threadsPerBlock>>>(_data);
}

void Sph::updateVelocities(float deltaTime) const
{
    kernel::updateVelocities<<<getBlocksPerGridForFluidParticles(), threadsPerBlock>>>(_data, deltaTime);
}

void Sph::updatePositions(float deltaTime) const
{
    kernel::updatePositions<<<getBlocksPerGridForFluidParticles(), threadsPerBlock>>>(_data, _domain, deltaTime);
}

void Sph::updateColors() const
{
    kernel::updateColors<<<getBlocksPerGridForFluidParticles(), threadsPerBlock>>>(_data);
}
}
