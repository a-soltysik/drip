#include "drip/engine/resource/Object.hpp"

#include <string>
#include <utility>
#include <vector>

#include "drip/engine/resource/Surface.hpp"

namespace drip::engine::gfx
{

auto Object::getId() const noexcept -> Id
{
    return _id;
}

Object::Object(std::string name)
    : _name {std::move(name)},
      _id {currentId++}
{
}

auto Object::getName() const noexcept -> const std::string&
{
    return _name;
}

void Object::addSurface(const Surface& surface)
{
    surfaces.push_back(surface);
}

auto Object::getSurfaces() const noexcept -> const std::vector<Surface>&
{
    return surfaces;
}

auto Object::getNextId() -> Id
{
    return currentId;
}
}
