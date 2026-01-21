#pragma once
#include <thrust/device_vector.h>

#include "SimulationParameters.cuh"
#include "Span.cuh"

namespace drip::sim
{
class NeighborGrid
{
public:
    struct Grid
    {
        glm::ivec3 gridSize;
        glm::vec3 cellSize;
    };

    struct KernelLaunchConfig
    {
        dim3 blocksPerGrid;
        dim3 threadsPerBlock;
    };

    struct DeviceView
    {
        struct ParticleIndexData
        {
            Span<int32_t> cellStartIndices;
            Span<int32_t> cellEndIndices;
            Span<int32_t> particleGridIndices;
            Span<int32_t> particleArrayIndices;

            uint32_t particleCount;
        };

        SimulationParameters::Domain domain;
        Grid grid;
        ParticleIndexData fluidIndexData;

        template <typename Func>
        __device__ void forEachFluidNeighbor(glm::vec4 position,
                                             const glm::vec4* positions,
                                             float smoothingRadius,
                                             Func&& func) const;

        __device__ auto calculateCellIndex(glm::vec4 position) const -> glm::ivec3;
        __device__ auto flattenCellIndex(glm::ivec3 cellIndex) const -> uint32_t;
    };

    NeighborGrid(const SimulationParameters::Domain& domain, float cellWidth, uint32_t fluidParticleCapacity);

    void update(const KernelLaunchConfig& runData, glm::vec4* positions, uint32_t fluidParticleCount);
    [[nodiscard]] auto toDeviceView() -> DeviceView;

private:
    struct ParticleIndexData
    {
        thrust::device_vector<int32_t> cellStartIndices;
        thrust::device_vector<int32_t> cellEndIndices;
        thrust::device_vector<int32_t> particleGridIndices;
        thrust::device_vector<int32_t> particleArrayIndices;

        uint32_t particleCount;
    };

    static auto createParticleIndexData(uint32_t totalCells, uint32_t particleCapacity) -> ParticleIndexData;
    static auto calculateGridSize(const SimulationParameters::Domain& domain, float cellWidth) -> glm::ivec3;
    static auto createGrid(const SimulationParameters::Domain& domain, float cellWidth) -> Grid;
    static auto getCellCount(const glm::ivec3& gridSize) -> uint32_t;
    static void resetGrid(ParticleIndexData& data);
    static void sortParticles(ParticleIndexData& data);
    static void assignParticlesToCells(const KernelLaunchConfig& runData, glm::vec4* positions);
    static void calculateCellStartAndEndIndices(const KernelLaunchConfig& runData);

    SimulationParameters::Domain _domain;
    Grid _grid;
    ParticleIndexData _fluidIndexData;
    float _cellWidth;
};

template <typename Func>
void __device__ NeighborGrid::DeviceView::forEachFluidNeighbor(glm::vec4 position,
                                                               const glm::vec4* positions,
                                                               float smoothingRadius,
                                                               Func&& func) const
{
    const auto min = glm::max(glm::ivec3 {(glm::vec3 {position} - domain.bounds.min - smoothingRadius) / grid.cellSize},
                              glm::ivec3 {0, 0, 0});

    const auto max = glm::min(glm::ivec3 {(glm::vec3 {position} - domain.bounds.min + smoothingRadius) / grid.cellSize},
                              grid.gridSize - 1);

    for (auto x = min.x; x <= max.x; x++)
    {
        for (auto y = min.y; y <= max.y; y++)
        {
            for (auto z = min.z; z <= max.z; z++)
            {
                const auto cellIdx = flattenCellIndex(glm::ivec3 {x, y, z});
                const auto startIdx = fluidIndexData.cellStartIndices[cellIdx];
                const auto endIdx = fluidIndexData.cellEndIndices[cellIdx];

                if (startIdx == -1 || startIdx > endIdx)
                {
                    continue;
                }

                for (auto i = startIdx; i <= endIdx; i++)
                {
                    const auto neighborIdx = fluidIndexData.particleArrayIndices[i];
                    const auto neighborPos = positions[neighborIdx];
                    func(neighborIdx, neighborPos);
                }
            }
        }
    }
}

inline __device__ auto NeighborGrid::DeviceView::calculateCellIndex(glm::vec4 position) const -> glm::ivec3
{
    const auto relativePosition = glm::vec3 {position} - domain.bounds.min;
    const auto clampedPosition = glm::clamp(relativePosition, glm::vec3(0.F), domain.bounds.max - domain.bounds.min);

    return glm::min(glm::ivec3 {clampedPosition / grid.cellSize}, grid.gridSize - 1);
}

inline __device__ auto NeighborGrid::DeviceView::flattenCellIndex(glm::ivec3 cellIndex) const -> uint32_t
{
    return cellIndex.x + (cellIndex.y * grid.gridSize.x) + (cellIndex.z * grid.gridSize.x * grid.gridSize.y);
}

}