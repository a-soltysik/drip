#pragma once

// clang-format off
#include <drip/common/utils/Assert.hpp> // NOLINT(misc-include-cleaner)
// clang-format on

#include <cstddef>
#include <glm/ext/vector_float3.hpp>
#include <string>
#include <vector>

#include "Surface.hpp"

namespace drip::engine::gfx
{

struct Transform
{
    glm::vec3 translation {};
    glm::vec3 scale {1.F, 1.F, 1.F};
    glm::vec3 rotation {};
};

class Scene;
class Context;

class Object
{
public:
    using Id = size_t;

    static auto getNextId() -> Id;

    explicit Object(std::string name);
    Object(const Object&) = delete;
    Object(Object&&) = delete;
    auto operator=(const Object&) = delete;
    auto operator=(Object&&) = delete;

    ~Object() noexcept = default;

    [[nodiscard]] auto getId() const noexcept -> Id;
    [[nodiscard]] auto getName() const noexcept -> const std::string&;
    void addSurface(const Surface& surface);
    [[nodiscard]] auto getSurfaces() const noexcept -> const std::vector<Surface>&;

    Transform transform;

private:
    inline static Id currentId = 0;
    std::vector<Surface> surfaces;
    std::string _name;
    Id _id;
};

}
