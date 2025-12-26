#pragma once

namespace drip::engine::gfx
{

struct FrameInfo;

class RenderSystem
{
public:
    RenderSystem() = default;
    RenderSystem(const RenderSystem&) = delete;
    RenderSystem& operator=(const RenderSystem&) = delete;
    RenderSystem(RenderSystem&&) = delete;
    RenderSystem& operator=(RenderSystem&&) = delete;
    virtual ~RenderSystem() noexcept = default;

    virtual void render(const FrameInfo& frameInfo) const = 0;
};

}
