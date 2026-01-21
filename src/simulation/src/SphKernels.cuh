#pragma once

#include "SimulationParameters.cuh"
#include "Sph.cuh"

namespace drip::sim
{
void uploadSimulationParameters(const SimulationParameters& parameters);

namespace kernel
{

__global__ void computeDensities(Sph::FluidParticlesData particles, NeighborGrid::DeviceView grid);
__global__ void computePressureAccelerations(Sph::FluidParticlesData particles, NeighborGrid::DeviceView grid);
__global__ void computeViscosityAccelerations(Sph::FluidParticlesData particles, NeighborGrid::DeviceView grid);
__global__ void computeSurfaceTensionAccelerations(Sph::FluidParticlesData particles, NeighborGrid::DeviceView grid);
__global__ void computeExternalAccelerations(Sph::FluidParticlesData particles);
__global__ void updateVelocities(Sph::FluidParticlesData particles, float dt);
__global__ void updatePositions(Sph::FluidParticlesData particles, float dt);
__global__ void updateColors(Sph::FluidParticlesData particles);

}
}