#pragma once
#include "NeighborGrid.cuh"

namespace drip::sim
{

void uploadNeighborGrid(const NeighborGrid::DeviceView& neighborGrid);

namespace kernel
{

__global__ void assignParticlesToCells(glm::vec4* positions);
__global__ void calculateCellStartAndEndIndices();

}
}