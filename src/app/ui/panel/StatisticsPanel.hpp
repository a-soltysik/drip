#pragma once

#include "drip/gfx/gui/GuiPanel.hpp"

namespace drip::app
{
class StatisticsPanel : public gfx::GuiPanel
{
public:
    void render() override;

private:
    static void renderFramerate();
};

}
