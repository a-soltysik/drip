#pragma once
#include "drip/engine/gui/GuiPanel.hpp"
#include "imgui.h"
#include "utils/IntervalCache.hpp"

namespace drip::app
{
class StatisticsPanel : public engine::GuiPanel
{
public:
    void render() override;

private:
    static void renderFramerate();
};

}
