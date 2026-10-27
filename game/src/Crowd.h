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
