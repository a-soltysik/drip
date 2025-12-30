#pragma once

namespace drip::engine::gfx
{

class Renderable
{
public:
    enum class Type
    {
        Mesh,
        Billboard,
    };

    virtual ~Renderable() = default;
    virtual auto getType() const -> Type = 0;
    virtual auto getName() const -> std::string_view = 0;
};

}
