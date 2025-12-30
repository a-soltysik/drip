#pragma once

#include <drip/engine/resource/Mesh.hpp>
#include <memory>
#include <string>

namespace drip
{
namespace engine
{
class Context;
}

namespace app::mesh::inverted_cube
{
auto create(const engine::gfx::Context& context, std::string name) -> std::unique_ptr<engine::gfx::Mesh>;
}
}