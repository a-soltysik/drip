#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

#include <drip/common/log/LogMessageBuilder.hpp>
#include <drip/common/utils/format/GlmFormatter.hpp>
#include <glm/gtx/string_cast.hpp>

#include "Sph.cuh"

namespace drip::sim
{

Sph::Sph(SharedMemory sharedMemory, Domain domain)
    : _sharedMemory(std::move(sharedMemory)),
      _data {.positions = _sharedMemory.positions->toSpan<glm::vec4>(),
             .colors = _sharedMemory.colors->toSpan<glm::vec4>(),
             .sizes = _sharedMemory.sizes->toSpan<float>()}
{
    common::log::Info("Particle grid: {}", domain.sampling);

    static constexpr auto spacing = 0.05F;
    const auto violet = glm::vec4 {0.5F, 0.F, 1.F, 1.F};
    const auto hostPositions = createParticlePositions(domain, _data.positions.size());
    thrust::copy_n(hostPositions.begin(),
                   _data.positions.size(),
                   thrust::device_ptr<glm::vec4>(_data.positions.data()));
    thrust::fill_n(thrust::device, _data.colors.data(), _data.colors.size(), violet);
    thrust::fill_n(thrust::device, _data.sizes.data(), _data.sizes.size(), spacing);

    common::log::Info("Sph simulation created with {} particles", _data.positions.size());
}

void Sph::update(float /*deltaTime*/) { }

auto Sph::createParticlePositions(Domain domain, size_t particleCount) -> thrust::host_vector<glm::vec4>
{
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
}
