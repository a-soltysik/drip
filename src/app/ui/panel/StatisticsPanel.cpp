#include "StatisticsPanel.hpp"

#include <imgui.h>

#include <chrono>

#include "utils/IntervalCache.hpp"

namespace drip::app
{

void StatisticsPanel::render()
{
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
    ImGui::Begin("Statistics", nullptr, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_AlwaysAutoResize);
    renderFramerate();
    ImGui::End();
}

void StatisticsPanel::renderFramerate()
{
    static auto fpsCache = utils::IntervalCache {std::chrono::seconds {1}, [] {
                                                     return ImGui::GetIO().Framerate;
                                                 }};

    //NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
    ImGui::Text("FPS: %.1f", static_cast<double>(fpsCache.get()));
}
}
