#include "Scene.hpp"

#include <drip/common/utils/Assert.hpp>
#include <drip/gfx/resource/MeshRenderable.hpp>
#include <drip/gfx/resource/ParticlesRenderable.hpp>
#include <drip/gfx/resource/Surface.hpp>
#include <drip/gfx/resource/Texture.hpp>
#include <drip/gfx/scene/Camera.hpp>
#include <drip/gfx/scene/Light.hpp>
#include <drip/gfx/scene/Scene.hpp>
#include <drip/gfx/vulkan/core/Context.hpp>
#include <glm/ext/vector_uint3.hpp>
#include <memory>
#include <string>
#include <utility>

#include "mesh/InvertedCube.hpp"

namespace drip::app::utils
{
auto Scene::createDefaultScene(gfx::Context& context) -> std::unique_ptr<Scene>
{
    auto scene = gfx::Scene {};
    auto blueTexture = gfx::Texture::getDefaultTexture(context, {0.25, 0.25, 0.3, 1.F});
    auto invertedCubeMesh = mesh::inverted_cube::create(context, "InvertedCube");
    auto domainRenderable = std::make_unique<gfx::MeshRenderable>(std::string {domainName});
    domainRenderable->addSurface(gfx::Surface {blueTexture.get(), invertedCubeMesh.get()});
    static constexpr auto domainSampling = glm::uvec3 {10};
    const auto domain =
        Domain {.min = domainRenderable->transform.translation - domainRenderable->transform.scale / 2.F,
                .max = domainRenderable->transform.translation + domainRenderable->transform.scale / 2.F,
                .sampling = domainSampling};

    scene.addRenderable(std::move(domainRenderable));

    context.registerMesh(std::move(invertedCubeMesh));
    context.registerTexture(std::move(blueTexture));

    scene.addRenderable(
        std::make_unique<gfx::ParticlesRenderable>(context,
                                                   std::string {fluidParticlesName},
                                                   domainSampling.x * domainSampling.y * domainSampling.z));

    auto directionalLight = gfx::DirectionalLight {};
    directionalLight.name = "DirectionalLight";
    directionalLight.direction = {-6.2F, -2.F, -1.F};
    directionalLight.makeColorLight({1.F, .8F, .8F}, 0.F, 0.8F, 1.F, 0.8F);

    scene.addLight(std::move(directionalLight));

    auto cameraObject = gfx::Transform {
        .translation = {0, 0.5, -5}
    };
    scene.getCamera().setViewYXZ(
        gfx::view::YXZ {.position = cameraObject.translation, .rotation = cameraObject.rotation});
    return std::unique_ptr<Scene> {
        new Scene {std::move(scene), domain}
    };
}

Scene::Scene(gfx::Scene scene, Domain domain)
    : _scene(std::move(scene)),
      _domain(domain)
{
}

auto Scene::getGfxScene() const -> const gfx::Scene&
{
    return _scene;
}

auto Scene::getGfxScene() -> gfx::Scene&
{
    return _scene;
}

auto Scene::getFluidParticles() const -> const gfx::ParticlesRenderable&
{
    const auto particles = common::Expect(_scene.findRenderableByName(fluidParticlesName),
                                          "{} renderable doesn't exist in the scene",
                                          fluidParticlesName)
                               .result();
    return dynamic_cast<const gfx::ParticlesRenderable&>(particles.get());
}

auto Scene::getDomain() const -> const Domain&
{
    return _domain;
}
}