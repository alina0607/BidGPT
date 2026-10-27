#pragma once

#include "sa/Backend.h"

namespace sa
{

// A texture cut into equal cells, numbered row by row from the top-left, like CS529's
// SpriteSource. `frames` can be less than cols × rows: trailing cells may be empty.
struct SpriteSheet
{
    TextureId texture = 0;
    int cols = 1;
    int rows = 1;
    int frames = 1;

    // Texture coordinates of one cell; frames outside 0..frames-1 are clamped.
    UVRect Cell(int frame) const;
    void Draw(Backend& backend, int frame, Vec2 topLeft, Vec2 size, float alpha = 1.0f) const;
};

// Steps through frames over time, like CS529's Animation::Play(count, delay, loop).
class Animation
{
public:
    void Play(int frameCount, float frameTime, bool loop);
    void Update(float dt) { time_ += dt; }

    int Frame() const;
    // A non-looping animation has shown its last frame for its full time.
    bool Done() const;

private:
    int count_ = 1;
    float frameTime_ = 0.0f;
    bool loop_ = false;
    float time_ = 0.0f;
};

} // namespace sa
