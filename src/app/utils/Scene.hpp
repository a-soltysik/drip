#pragma once

#include <drip/gfx/resource/ParticlesRenderable.hpp>
#include <drip/gfx/scene/Scene.hpp>

namespace drip::app::utils
{
class Scene
{
public:
    struct Domain
    {
        glm::vec3 min;
        glm::vec3 max;
        glm::uvec3 sampling;
    };

    static auto createDefaultScene(gfx::Context& context) -> std::unique_ptr<Scene>;

    [[nodiscard]] auto getGfxScene() const -> const gfx::Scene&;
    [[nodiscard]] auto getGfxScene() -> gfx::Scene&;
    [[nodiscard]] auto getDomain() const -> const Domain&;
    [[nodiscard]] auto getFluidParticles() const -> const gfx::ParticlesRenderable&;

private:
    static constexpr auto fluidParticlesName = std::string_view {"FluidParticles"};
    static constexpr auto domainName = std::string_view {"Domain"};

    Scene(gfx::Scene scene, Domain domain);

    gfx::Scene _scene;
    Domain _domain;
};
}
