#include "sa/Sprite.h"

#include <algorithm>

namespace sa
{

UVRect SpriteSheet::Cell(int frame) const
{
    const int f = std::clamp(frame, 0, std::max(frames, 1) - 1);
    const Vec2 size{1.0f / static_cast<float>(cols), 1.0f / static_cast<float>(rows)};
    return {{static_cast<float>(f % cols) * size.x, static_cast<float>(f / cols) * size.y}, size};
}

void SpriteSheet::Draw(Backend& backend, int frame, Vec2 topLeft, Vec2 size, float alpha) const
{
    backend.DrawSprite(texture, Cell(frame), topLeft, size, alpha);
}

void Animation::Play(int frameCount, float frameTime, bool loop)
{
    count_ = std::max(frameCount, 1);
    frameTime_ = frameTime;
    loop_ = loop;
    time_ = 0.0f;
}

int Animation::Frame() const
{
    if (frameTime_ <= 0.0f)
        return 0;
    const int step = static_cast<int>(time_ / frameTime_);
    return loop_ ? step % count_ : std::min(step, count_ - 1);
}

bool Animation::Done() const { return !loop_ && time_ >= frameTime_ * static_cast<float>(count_); }

} // namespace sa
