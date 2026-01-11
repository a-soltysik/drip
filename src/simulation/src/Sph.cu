#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include <drip/common/log/LogMessageBuilder.hpp>
#include <glm/ext/vector_float4.hpp>

#include "Sph.cuh"

namespace drip::sim
{

Sph::Sph(SharedMemory sharedMemory)
    : _sharedMemory(std::move(sharedMemory)),
      _data {.positions = _sharedMemory.positions->toSpan<glm::vec4>(),
             .colors = _sharedMemory.colors->toSpan<glm::vec4>(),
             .sizes = _sharedMemory.sizes->toSpan<float>()}
{
    const auto violet = glm::vec4 {0.5F, 0.F, 1.F, 1.F};

    common::log::Info("Sph simulation initializing with {} particles", _data.positions.size());

    thrust::fill_n(thrust::device, _data.positions.data(), _data.positions.size(), glm::vec4 {0.0f});
    thrust::fill_n(thrust::device, _data.colors.data(), _data.colors.size(), violet);
    thrust::fill_n(thrust::device, _data.sizes.data(), _data.sizes.size(), 0.1F);

    common::log::Info("Sph simulation created with {} particles", _data.positions.size());
}

void Sph::update(float /*deltaTime*/) { }
}
