#pragma once

namespace drip::sim::utils
{

inline __device__ auto turboColormap(float value) -> glm::vec3
{
    const auto redVec4 = glm::vec4 {0.13572138, 4.61539260, -42.66032258, 132.13108234};
    const auto greenVec4 = glm::vec4 {0.09140261, 2.19418839, 4.84296658, -14.18503333};
    const auto blueVec4 = glm::vec4 {0.10667330, 12.64194608, -60.58204836, 110.36276771};
    const auto redVec2 = glm::vec2 {-152.94239396, 59.28637943};
    const auto greenVec2 = glm::vec2 {4.27729857, 2.82956604};
    const auto blueVec2 = glm::vec2 {-89.90310912, 27.34824973};

    value = glm::clamp(value, 0.F, 1.F);
    const auto v4 = glm::vec4 {1.0F, value, value * value, value * value * value};
    const auto v2 = glm::vec2 {v4.z, v4.w} * v4.z;
    return {glm::dot(v4, redVec4) + glm::dot(v2, redVec2),
            glm::dot(v4, greenVec4) + glm::dot(v2, greenVec2),
            glm::dot(v4, blueVec4) + glm::dot(v2, blueVec2)};
}

}