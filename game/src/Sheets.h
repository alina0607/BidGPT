#pragma once

#include "sa/Backend.h"
#include "sa/Sprite.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

// A one-row sheet from scripts/make_stage_art.py: cells, the feet's place in a cell, how long each frame
// shows, how tall the standing figure is in texture pixels, and named runs of frames.
struct Sheet
{
    sa::SpriteSheet sheet;
    sa::Vec2 cell;
    sa::Vec2 anchor;
    std::vector<float> seconds;
    float standing = 1.0f;
    std::map<std::string, std::pair<int, int>> parts;   // first and last frame, inclusive
    std::map<std::string, int> poses;                    // single frames by name (the "jump" ones)

    bool Loaded() const { return sheet.texture != 0; }
    float Length(int first, int last) const;
    // The frame of first..last shown `t` seconds in; looping, or holding the last frame.
    int FrameAt(float t, int first, int last, bool loop) const;
    // A named part's frames; the whole sheet if it has no such part.
    std::pair<int, int> Part(const std::string& name) const;
};

// <dir>/<name>.png and its .json; an empty Sheet (Loaded() false) if either is missing.
Sheet LoadSheet(sa::Backend& backend, const std::string& dir, const std::string& name);

// One frame with its feet at `feet` on screen, `scale` times its texture size; mirrored, it faces left.
void DrawFrame(sa::Backend& backend, const Sheet& s, int frame, sa::Vec2 feet, float scale, bool mirrored = false,
               float alpha = 1.0f);
