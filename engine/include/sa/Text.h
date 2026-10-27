#pragma once

#include "sa/Backend.h"

#include <string>
#include <string_view>
#include <vector>

namespace sa
{

// Text is drawn in an 8×8 bitmap font (ASCII) with the backend's rectangles, so it works on every backend,
// until LoadFont gives a proportional font. Either way `scale` means the same: the text's letters fill a
// box kGlyphSize × scale high, and lines are LineHeight(scale) apart, so layouts work with both.
constexpr float kGlyphSize = 8.0f;
constexpr float kLineSpacing = 1.35f; // line height in glyph sizes

// A font atlas made by scripts/make_stage_art.py (a white-on-transparent PNG and its metrics file). False,
// and the bitmap font stays, if either can't be read.
bool LoadFont(Backend& backend, const std::string& atlasPath, const std::string& metricsPath);
// A second atlas in the same format for characters beyond ASCII (UTF-8 text, e.g. Chinese), used for any
// character LoadFont's font does not have. Without it, or for a character it lacks, '?' is drawn.
bool LoadFallbackFont(Backend& backend, const std::string& atlasPath, const std::string& metricsPath);

float TextWidth(std::string_view text, float scale);
float LineHeight(float scale);

// Draws one line of UTF-8 text; a character no loaded font has is drawn as '?'.
void DrawText(Backend& backend, Vec2 topLeft, std::string_view text, float scale, Color color);

// Splits text into lines no wider than maxWidth, breaking at spaces where possible, and between any two
// characters beyond ASCII (Chinese is written without spaces).
std::vector<std::string> WrapText(std::string_view text, float maxWidth, float scale);

// Draws wrapped text; returns the height used.
float DrawWrapped(Backend& backend, Vec2 topLeft, std::string_view text, float maxWidth, float scale,
                  Color color);

} // namespace sa
