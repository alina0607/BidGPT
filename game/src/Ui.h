#pragma once

#include "Stage.h"

#include "sa/Backend.h"
#include "sa/Text.h"

#include <algorithm>
#include <string>

namespace ui
{

// The market palette (docs/art.md): warm paper panels, brown ink, pink buttons.
inline constexpr sa::Color kBackground{0.96f, 0.95f, 0.92f, 1.0f};
inline constexpr sa::Color kInk{0.32f, 0.20f, 0.18f, 1.0f};
inline constexpr sa::Color kMuted{0.56f, 0.42f, 0.37f, 1.0f};
inline constexpr sa::Color kPanel{1.0f, 0.98f, 0.957f, 0.94f};
inline constexpr sa::Color kBorder{0.50f, 0.34f, 0.29f, 1.0f};
inline constexpr sa::Color kAccent{0.80f, 0.33f, 0.45f, 1.0f};       // the player's lines, the cursor
inline constexpr sa::Color kButton{0.93f, 0.56f, 0.64f, 1.0f};
inline constexpr sa::Color kButtonHover{0.84f, 0.42f, 0.52f, 1.0f};
inline constexpr sa::Color kButtonOff{0.80f, 0.74f, 0.71f, 1.0f};
inline constexpr sa::Color kGood{0.16f, 0.52f, 0.30f, 1.0f};
inline constexpr sa::Color kBad{0.78f, 0.24f, 0.24f, 1.0f};
inline constexpr sa::Color kWhite{1.0f, 1.0f, 1.0f, 1.0f};

enum class ButtonStyle
{
    Pink,
    Cream,
    Green,
};

struct Button
{
    sa::Vec2 pos;
    sa::Vec2 size;
    std::string label;
    bool enabled = true;
    ButtonStyle style = ButtonStyle::Pink;

    bool Contains(sa::Vec2 p) const
    {
        return p.x >= pos.x && p.x <= pos.x + size.x && p.y >= pos.y && p.y <= pos.y + size.y;
    }
};

// Turns "mouse is down" into "mouse was clicked this frame". Call Update once per frame.
class Mouse
{
public:
    void Update(sa::Backend& backend)
    {
        const bool down = backend.MouseDown(sa::MouseButton::Left);
        clicked_ = down && !wasDown_;
        wasDown_ = down;
        position_ = backend.MousePosition();
    }
    bool Clicked(const Button& b) const { return clicked_ && b.enabled && b.Contains(position_); }
    bool JustPressed() const { return clicked_; }
    sa::Vec2 Position() const { return position_; }

private:
    bool wasDown_ = true;   // ignore a click that is still held from the previous scene
    bool clicked_ = false;
    sa::Vec2 position_{};
};

inline void Panel(sa::Backend& b, sa::Vec2 pos, sa::Vec2 size, sa::Color fill = kPanel)
{
    if (stage::DrawUiSkin(b, stage::UiSkin::Panel, pos, size))
        return;
    if (stage::DrawRounded(b, pos, size, fill, kBorder))
        return;
    b.DrawRect({pos.x - 1, pos.y - 1}, {size.x + 2, size.y + 2}, kBorder);
    b.DrawRect(pos, size, fill);
}

// The panel size that holds `content` clear of the frame and flowers, and where content starts in it.
inline sa::Vec2 PanelFit(sa::Backend& b, sa::Vec2 content) { return stage::UiSkinFit(b, stage::UiSkin::Panel, content); }
inline sa::Vec2 PanelInset(sa::Backend& b, sa::Vec2 size)
{
    const auto in = stage::UiSkinInsets(b, stage::UiSkin::Panel, size);
    return {in.left, in.top};
}

// A screen's title, on a small panel in the top-left corner.
inline void Title(sa::Backend& b, const std::string& text)
{
    const float scale = 2.5f;
    const float width = std::max(360.0f, sa::TextWidth(text, scale) + 100);
    const float height = 64;
    if (!stage::DrawUiSkin(b, stage::UiSkin::Title, {14, 8}, {width, height}))
        Panel(b, {14, 8}, {width, height});
    sa::DrawText(b, {62, 8 + (height - sa::kGlyphSize * scale) / 2}, text, scale, kInk);
}

inline void DrawButton(sa::Backend& b, const Button& button, sa::Vec2 mouse, float scale = 2.0f)
{
    const bool hovered = button.enabled && button.Contains(mouse);
    const float alpha = !button.enabled ? 0.55f : hovered ? 0.86f : 1.0f;
    const stage::UiSkin skin = button.style == ButtonStyle::Green ? stage::UiSkin::ButtonGreen
                                 : button.style == ButtonStyle::Cream ? stage::UiSkin::ButtonCream
                                                                      : stage::UiSkin::Button;
    if (!stage::DrawUiSkin(b, skin, button.pos, button.size, alpha))
    {
        const sa::Color fill = !button.enabled ? kButtonOff : hovered ? kButtonHover : kButton;
        if (!stage::DrawRounded(b, button.pos, button.size, fill, kButtonHover))
            b.DrawRect(button.pos, button.size, fill);
    }
    const float textHeight = sa::kGlyphSize * scale;
    const float maxWidth = button.size.x - 28;
    std::string label = button.label;
    if (sa::TextWidth(label, scale) > maxWidth)
    {
        while (!label.empty() && sa::TextWidth(label + "...", scale) > maxWidth)
            label.pop_back();
        label += "...";
    }
    const float x = button.pos.x + (button.size.x - sa::TextWidth(label, scale)) / 2;
    sa::DrawText(b, {x, button.pos.y + (button.size.y - textHeight) / 2}, label, scale,
                 button.enabled ? kInk : kMuted);
}

inline void SpeechPanel(sa::Backend& b, sa::Vec2 pos, sa::Vec2 size, bool question = false)
{
    const sa::Color tint = question ? sa::Color{0.78f, 0.92f, 0.70f, 1.0f} : kWhite;
    if (!stage::DrawUiSkin(b, stage::UiSkin::Speech, pos, size, 1.0f, tint))
        Panel(b, pos, size);
}

} // namespace ui
