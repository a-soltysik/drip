#pragma once

#include "Sph.cuh"

namespace drip::sim::kernel
{

__global__ void computeDensities(Sph::FluidParticlesData particles);
__global__ void computePressureAccelerations(Sph::FluidParticlesData particles);
__global__ void computeViscosityAccelerations(Sph::FluidParticlesData particles);
__global__ void computeSurfaceTensionAccelerations(Sph::FluidParticlesData particles);
__global__ void computeExternalAccelerations(Sph::FluidParticlesData particles);
__global__ void updateVelocities(Sph::FluidParticlesData particles, float dt);
__global__ void updatePositions(Sph::FluidParticlesData particles, Simulation::Domain domain, float dt);
__global__ void updateColors(Sph::FluidParticlesData particles);

}