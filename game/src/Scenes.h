#pragma once

#include "GameContext.h"

#include "sa/Engine.h"

#include <nlohmann/json.hpp>

#include <memory>
#include <string>

// Draw → confirm what it is → sell it to the buyers → back to drawing. Or buy a cat from the model seller.
constexpr float kWidth = 1280.0f;
constexpr float kHeight = 720.0f;

std::unique_ptr<sa::Scene> MakeCanvasScene(GameContext& ctx);
std::unique_ptr<sa::Scene> MakeConfirmScene(GameContext& ctx);
std::unique_ptr<sa::Scene> MakeRoundScene(GameContext& ctx, nlohmann::json firstState);
std::unique_ptr<sa::Scene> MakeBuyScene(GameContext& ctx);
// Two buyer models side by side on one cat sale (--compare b4f9,b5): every buyer's line from each model.
std::unique_ptr<sa::Scene> MakeCompareScene(GameContext& ctx, std::string models);

// The message of an error reply. With --autoplay it also ends the run: a test that hits an
// error should fail at once with the reason, not wait until it runs out of frames.
std::string ReplyError(GameContext& ctx, sa::Engine& engine, const nlohmann::json& reply);

// Draws strokes scaled to fit a box (used for the thumbnail of the player's drawing).
void DrawStrokesInBox(sa::Backend& b, const std::vector<Stroke>& strokes, sa::Vec2 pos, sa::Vec2 size,
                      float thickness, sa::Color color);
