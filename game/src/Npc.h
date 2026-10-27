#pragma once

#include "Sheets.h"

#include "sa/Backend.h"

#include <cstddef>

// A buyer drawn as a painted NPC (assets/stage/npc_<n>): walks up from the flowers at the bottom of the screen,
// seen from behind, to their place in front of the stall; when they lose interest, turns round and walks back
// out through the flowers towards the camera. The n-th buyer gets npc_n while there are drawings left; the
// rest stay pixel sprites (Characters.h). Each painted buyer has a separate planted back-facing idle asset,
// so arriving never freezes a walking frame or turns around to face the player.
class Walker
{
public:
    Walker(size_t look, float delay) : look_(look), wait_(delay) {}

    void Update(float dt, bool present);
    // Feet at `spot` (screen) once arrived; below it, a little larger, while walking in or out.
    void Draw(sa::Backend& backend, sa::Vec2 spot) const;
    bool Gone() const { return phase_ == Phase::Gone; }
    // Standing at the stall, or gone: their tag shows (not while they walk through where it is).
    bool Settled() const { return phase_ == Phase::Standing || phase_ == Phase::Gone; }

private:
    enum class Phase
    {
        Waiting,
        Coming,
        Standing,
        Turning,
        Going,
        Gone,
    };

    size_t look_;
    float wait_;
    Phase phase_ = Phase::Waiting;
    float time_ = 0.0f;
    float below_ = 0.0f;        // how far below the spot the feet are
};

namespace npcs
{

void Load(sa::Backend& backend);
size_t Count();

} // namespace npcs
