#include "NeighborKernels.cuh"

namespace drip::sim
{

namespace kernel::constant
{
__constant__ NeighborGrid::DeviceView neighborGrid;
}

void uploadNeighborGrid(const NeighborGrid::DeviceView& neighborGrid)
{
    cudaMemcpyToSymbol(kernel::constant::neighborGrid, &neighborGrid, sizeof(NeighborGrid::DeviceView));
}

namespace kernel
{
__global__ void assignParticlesToCells(glm::vec4* positions)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < constant::neighborGrid.fluidIndexData.particleCount)
    {
        constant::neighborGrid.fluidIndexData.particleArrayIndices[idx] = idx;
        const auto cellPosition = constant::neighborGrid.calculateCellIndex(positions[idx]);
        const auto cellIndex = constant::neighborGrid.flattenCellIndex(cellPosition);
        constant::neighborGrid.fluidIndexData.particleGridIndices[idx] = cellIndex;
    }
}

__global__ void calculateCellStartAndEndIndices()
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= constant::neighborGrid.fluidIndexData.particleCount)
    {
        return;
    }

    const auto cellIdx = constant::neighborGrid.fluidIndexData.particleGridIndices[idx];
    const auto particleCount = constant::neighborGrid.fluidIndexData.particleCount;

    if (idx == 0)
    {
        constant::neighborGrid.fluidIndexData.cellStartIndices[cellIdx] = idx;
    }
    else
    {
        const auto prevCellIdx = constant::neighborGrid.fluidIndexData.particleGridIndices[idx - 1];
        if (cellIdx != prevCellIdx)
        {
            constant::neighborGrid.fluidIndexData.cellStartIndices[cellIdx] = idx;
        }
    }
    if (idx == particleCount - 1)
    {
        constant::neighborGrid.fluidIndexData.cellEndIndices[cellIdx] = idx;
    }
    else
    {
        const auto nextCellIdx = constant::neighborGrid.fluidIndexData.particleGridIndices[idx + 1];
        if (cellIdx != nextCellIdx)
        {
            constant::neighborGrid.fluidIndexData.cellEndIndices[cellIdx] = idx;
        }
    }
}

}
}