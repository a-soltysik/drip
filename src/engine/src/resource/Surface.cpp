#include "drip/engine/resource/Surface.hpp"

#include "drip/engine/resource/Mesh.hpp"
#include "drip/engine/resource/Texture.hpp"

namespace drip::engine::gfx
{

Surface::Surface(const Texture* texture, const Mesh* mesh)
    : _texture {texture},
      _mesh {mesh}
{
}

auto Surface::getTexture() const noexcept -> const Texture&
{
    return *_texture;
}

auto Surface::getMesh() const noexcept -> const Mesh&
{
    return *_mesh;
}

}
