#pragma once

#include "SimulationParameters.cuh"
#include "Sph.cuh"

namespace drip::sim
{

void uploadFluidParticlesData(const Sph::FluidParticlesData& data);

namespace kernel
{

__global__ void computeDensities(SimulationParameters simulationParameters, NeighborGrid::DeviceView grid);
__global__ void computePressureAccelerations(SimulationParameters simulationParameters, NeighborGrid::DeviceView grid);
__global__ void computeViscosityAccelerations(SimulationParameters simulationParameters, NeighborGrid::DeviceView grid);
__global__ void computeSurfaceTensionAccelerations(SimulationParameters simulationParameters,
                                                   NeighborGrid::DeviceView grid);
__global__ void computeExternalAccelerations(SimulationParameters simulationParameters);
__global__ void updateVelocities(SimulationParameters simulationParameters, float dt);
__global__ void updatePositions(SimulationParameters simulationParameters, float dt);
__global__ void updateColors(SimulationParameters simulationParameters);

}
}