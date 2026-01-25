#pragma once
#include "NeighborGrid.cuh"

namespace drip::sim
{

namespace kernel
{

__global__ void assignParticlesToCells(NeighborGrid::DeviceView neighborGrid, const glm::vec4* positions);
__global__ void calculateCellStartAndEndIndices(NeighborGrid::DeviceView neighborGrid);

}
}