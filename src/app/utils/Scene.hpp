#pragma once

#include <drip/gfx/resource/ParticlesRenderable.hpp>
#include <drip/gfx/scene/Scene.hpp>
#include <drip/gfx/vulkan/core/Context.hpp>
#include <drip/simulation/SimulationConfig.cuh>
#include <memory>
#include <string_view>

namespace drip::app::utils
{
class Scene
{
public:
    static auto createDefaultScene(gfx::Context& context, const sim::SimulationConfig& simulationParameters)
        -> std::unique_ptr<Scene>;

    [[nodiscard]] auto getGfxScene() const -> const gfx::Scene&;
    [[nodiscard]] auto getGfxScene() -> gfx::Scene&;
    [[nodiscard]] auto getFluidParticles() const -> const gfx::ParticlesRenderable&;

private:
    static constexpr auto fluidParticlesName = std::string_view {"FluidParticles"};
    static constexpr auto domainName = std::string_view {"Domain"};

    explicit Scene(gfx::Scene scene);

    gfx::Scene _scene;
};
}
