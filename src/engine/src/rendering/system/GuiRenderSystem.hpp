#pragma once

#include "RenderSystem.hpp"
#include "drip/engine/gui/GuiManager.hpp"

namespace drip::engine::gfx
{
class GuiRenderSystem : public RenderSystem
{
public:
    explicit GuiRenderSystem(GuiManager& guiManager);
    void render(const FrameInfo& frameInfo) override;

private:
    GuiManager& _guiManager;
};
}
