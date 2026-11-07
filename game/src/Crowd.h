#pragma once

#include "Characters.h"
#include "Npc.h"
#include "Ui.h"

#include <nlohmann/json.hpp>

#include <map>
#include <string>
#include <vector>

// The buyers standing in a row in front of the counter, each under a tag with their name and offer.
// Hovering a buyer shows what they care about. Every buyer's latest line stays in a nearby bubble;
// the active question is tinted differently.

// "$4.50", "$1,325"; with cents = false always whole dollars ("$3").
std::string Money(double v, bool cents = true);

// Wraps text to `width` at `scale` in at most `maxLines` lines; if it doesn't fit, the last line ends in
// "..." so the reader knows it goes on, instead of stopping mid-sentence.
std::vector<std::string> WrapLines(const std::string& text, float width, size_t maxLines, float scale = 1.5f);

struct CrowdSpot
{
    std::string name;
    sa::Vec2 feet;
    ui::Button hit;   // the buyer and their tag: hover for details, click for their scene
};

std::vector<CrowdSpot> CrowdSpots(const nlohmann::json& buyers);

struct CrowdView
{
    const nlohmann::json* buyers = nullptr;
    const CharacterArt* art = nullptr;
    const std::map<std::string, BuyerSprite>* sprites = nullptr;
    const std::map<std::string, Walker>* walkers = nullptr;   // buyers drawn as painted NPCs
    sa::Vec2 mouse;
    const std::map<std::string, std::string>* speech = nullptr; // each buyer's latest line
    std::string asker;     // active question's buyer; their bubble uses the question tint
    bool scenes = false;   // whether a click opens a buyer's scene (said in the hover card)
    size_t speechLines = 3;   // the most lines a bubble shows before "..."
};

void DrawCrowd(sa::Backend& b, const CrowdView& view);

// "B4f9" for a buyer played by runs/sft_b4f9 ("buyer_gpt_b4f9"); empty when the state names no model.
std::string ModelLabel(const nlohmann::json& buyer);

// Who each buyer is, always in view (owner, 2026-10-28): name and model, persona, what they care about, in a panel
// `width` wide from `top`, the oldest buyers first, leaving out what does not fit above `bottom`. Returns its bottom.
float DrawBuyerCards(sa::Backend& b, const nlohmann::json& buyers, sa::Vec2 pos, float width, float bottom);

// One frame of the buyers' figures (loading the art the first time; autoplay prints how many looks it has):
// the first npcs::Count() buyers walk in as painted NPCs, a moment apart; the rest play their mood's animation.
void UpdateCrowdSprites(sa::Backend& b, CharacterArt& art, const nlohmann::json& buyers,
                        std::map<std::string, BuyerSprite>& sprites, std::map<std::string, Walker>& walkers, float dt,
                        bool autoplay);
