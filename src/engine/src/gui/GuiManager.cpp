#include "drip/engine/gui/GuiManager.hpp"

#include <algorithm>

#include "drip/engine/gui/GuiPanel.hpp"

namespace drip::engine::gfx
{

void GuiManager::render()
{
    std::ranges::for_each(_panels, &GuiPanel::render);
}

}
