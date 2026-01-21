#include "NeighborKernels.cuh"

namespace drip::sim::kernel
{

__global__ void assignParticlesToCells(NeighborGrid::DeviceView neighborGrid, const glm::vec4* positions)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < neighborGrid.fluidIndexData.particleCount)
    {
        neighborGrid.fluidIndexData.particleArrayIndices[idx] = idx;
        const auto cellPosition = neighborGrid.calculateCellIndex(positions[idx]);
        const auto cellIndex = neighborGrid.flattenCellIndex(cellPosition);
        neighborGrid.fluidIndexData.particleGridIndices[idx] = cellIndex;
    }
}

__global__ void calculateCellStartAndEndIndices(NeighborGrid::DeviceView neighborGrid)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= neighborGrid.fluidIndexData.particleCount)
    {
        return;
    }

    const auto cellIdx = neighborGrid.fluidIndexData.particleGridIndices[idx];
    const auto particleCount = neighborGrid.fluidIndexData.particleCount;

    if (idx == 0)
    {
        neighborGrid.fluidIndexData.cellStartIndices[cellIdx] = idx;
    }
    else
    {
        const auto prevCellIdx = neighborGrid.fluidIndexData.particleGridIndices[idx - 1];
        if (cellIdx != prevCellIdx)
        {
            neighborGrid.fluidIndexData.cellStartIndices[cellIdx] = idx;
        }
    }
    if (idx == particleCount - 1)
    {
        neighborGrid.fluidIndexData.cellEndIndices[cellIdx] = idx;
    }
    else
    {
        const auto nextCellIdx = neighborGrid.fluidIndexData.particleGridIndices[idx + 1];
        if (cellIdx != nextCellIdx)
        {
            neighborGrid.fluidIndexData.cellEndIndices[cellIdx] = idx;
        }
    }
}

}
