#pragma once

#include "Api.cuh"

namespace drip::sim
{

class DRIP_CUDA_API Simulation
{
public:
    void update(float deltaTime);
};

}
