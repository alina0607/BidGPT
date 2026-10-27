#include "Sheets.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>

float Sheet::Length(int first, int last) const
{
    float total = 0.0f;
    for (int i = first; i <= last && i < static_cast<int>(seconds.size()); ++i)
        total += seconds[i];
    return total;
}

int Sheet::FrameAt(float t, int first, int last, bool loop) const
{
    const float total = Length(first, last);
    if (total <= 0.0f)
        return first;
    t = loop ? std::fmod(t, total) : std::min(t, total - 1e-4f);
    for (int i = first; i <= last; ++i)
    {
        if (t < seconds[i])
            return i;
        t -= seconds[i];
    }
    return last;
}

std::pair<int, int> Sheet::Part(const std::string& name) const
{
    const auto it = parts.find(name);
    return it != parts.end() ? it->second : std::pair<int, int>{0, sheet.frames - 1};
}

Sheet LoadSheet(sa::Backend& backend, const std::string& dir, const std::string& name)
{
    Sheet s;
    std::ifstream in(dir + "/" + name + ".json");
    const auto meta = in ? nlohmann::json::parse(in, nullptr, false) : nlohmann::json();
    if (!meta.is_object())
        return s;
    s.sheet.texture = backend.LoadTexture(dir + "/" + name + ".png", sa::TextureFilter::Smooth);
    s.sheet.frames = meta.value("frames", 1);
    s.sheet.cols = meta.value("cols", s.sheet.frames);
    s.sheet.rows = meta.value("rows", 1);
    s.cell = {meta["cell"][0].get<float>(), meta["cell"][1].get<float>()};
    s.anchor = {meta["anchor"][0].get<float>(), meta["anchor"][1].get<float>()};
    s.standing = meta.value("standing", 1.0f);
    for (const auto& ms : meta.value("frame_ms", nlohmann::json::array()))
        s.seconds.push_back(ms.get<float>() / 1000.0f);
    const nlohmann::json parts = meta.value("parts", nlohmann::json::object());
    for (const auto& [part, range] : parts.items())
        s.parts[part] = {range[0].get<int>(), range[1].get<int>()};
    const nlohmann::json poses = meta.value("jump", nlohmann::json::object());
    for (const auto& [pose, frame] : poses.items())
        s.poses[pose] = frame.get<int>();
    return s;
}

void DrawFrame(sa::Backend& backend, const Sheet& s, int frame, sa::Vec2 feet, float scale, bool mirrored, float alpha)
{
    if (!s.Loaded())
        return;
    sa::UVRect uv = s.sheet.Cell(frame);
    const float anchorX = mirrored ? s.cell.x - s.anchor.x : s.anchor.x;
    if (mirrored)
    {
        uv.offset.x += uv.size.x;
        uv.size.x = -uv.size.x;
    }
    backend.DrawSprite(s.sheet.texture, uv, {feet.x - anchorX * scale, feet.y - s.anchor.y * scale},
                       {s.cell.x * scale, s.cell.y * scale}, alpha);
}
