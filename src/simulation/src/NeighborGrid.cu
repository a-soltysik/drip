#include <thrust/fill.h>
#include <thrust/sort.h>

#include "NeighborGrid.cuh"
#include "NeighborKernels.cuh"

namespace drip::sim
{
NeighborGrid::NeighborGrid(const SimulationParameters::Domain& domain, float cellWidth, uint32_t fluidParticleCapacity)
    : _domain {domain},
      _grid {createGrid(domain, cellWidth)},
      _fluidIndexData {createParticleIndexData(getCellCount(_grid.gridSize), fluidParticleCapacity)},
      _cellWidth {cellWidth}
{
}

void NeighborGrid::update(const KernelLaunchConfig& runData, glm::vec4* positions, uint32_t fluidParticleCount)
{
    _fluidIndexData.particleCount = fluidParticleCount;
    resetGrid(_fluidIndexData);
    assignParticlesToCells(runData, positions);
    sortParticles(_fluidIndexData);
    calculateCellStartAndEndIndices(runData);
}

auto NeighborGrid::createParticleIndexData(uint32_t totalCells, uint32_t particleCapacity) -> ParticleIndexData
{
    return {.cellStartIndices = thrust::device_vector<int32_t>(totalCells),
            .cellEndIndices = thrust::device_vector<int32_t>(totalCells),
            .particleGridIndices = thrust::device_vector<int32_t>(particleCapacity),
            .particleArrayIndices = thrust::device_vector<int32_t>(particleCapacity),
            .particleCount = 0};
}

auto NeighborGrid::calculateGridSize(const SimulationParameters::Domain& domain, float cellWidth) -> glm::ivec3
{
    return glm::ivec3 {glm::ceil(domain.getSize() / cellWidth)};
}

auto NeighborGrid::createGrid(const SimulationParameters::Domain& domain, float cellWidth) -> Grid
{
    const auto gridSize = calculateGridSize(domain, cellWidth);
    const auto cellSize = domain.getSize() / glm::vec3 {gridSize};
    return {
        .gridSize = gridSize,
        .cellSize = cellSize,
    };
}

auto NeighborGrid::getCellCount(const glm::ivec3& gridSize) -> uint32_t
{
    return static_cast<uint32_t>(gridSize.x * gridSize.y * gridSize.z);
}

void NeighborGrid::resetGrid(ParticleIndexData& data)
{
    thrust::fill(data.cellStartIndices.begin(), data.cellStartIndices.end(), -1);
    thrust::fill(data.cellEndIndices.begin(), data.cellEndIndices.end(), -1);
}

void NeighborGrid::assignParticlesToCells(const KernelLaunchConfig& runData, glm::vec4* positions)
{
    kernel::assignParticlesToCells<<<runData.blocksPerGrid, runData.threadsPerBlock>>>(toDeviceView(), positions);
}

void NeighborGrid::sortParticles(ParticleIndexData& data)
{
    thrust::sort_by_key(data.particleGridIndices.begin(),
                        data.particleGridIndices.end(),
                        data.particleArrayIndices.begin());
}

void NeighborGrid::calculateCellStartAndEndIndices(const KernelLaunchConfig& runData)
{
    kernel::calculateCellStartAndEndIndices<<<runData.blocksPerGrid, runData.threadsPerBlock>>>(toDeviceView());
}

auto NeighborGrid::toDeviceView() -> DeviceView
{
    return {
        .domain = _domain,
        .grid = _grid,
        .fluidIndexData = {
                           .cellStartIndices = Span<int32_t>::fromDeviceVector(_fluidIndexData.cellStartIndices),
                           .cellEndIndices = Span<int32_t>::fromDeviceVector(_fluidIndexData.cellEndIndices),
                           .particleGridIndices = Span<int32_t>::fromDeviceVector(_fluidIndexData.particleGridIndices),
                           .particleArrayIndices = Span<int32_t>::fromDeviceVector(_fluidIndexData.particleArrayIndices),
                           .particleCount = _fluidIndexData.particleCount}
    };
}
}