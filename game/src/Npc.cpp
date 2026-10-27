#include "Npc.h"

#include "Characters.h"
#include "Stage.h"

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

namespace
{

constexpr float kStartBelow = 130.0f;  // half the old entrance distance; starts just below the flower bed
constexpr float kWalkSpeed = 110.0f;   // px/s

struct NpcLook
{
    Sheet walk;
    Sheet idle;
};

std::vector<NpcLook> looks;
bool tried = false;

} // namespace

namespace npcs
{

void Load(sa::Backend& backend)
{
    if (tried)
        return;
    tried = true;
    const std::string dir = AssetDir() + "/stage";
    for (int n = 1; std::ifstream(dir + "/npc_" + std::to_string(n) + ".json"); ++n)
    {
        const std::string name = "npc_" + std::to_string(n);
        looks.push_back({LoadSheet(backend, dir, name), LoadSheet(backend, dir, name + "_idle")});
    }
}

size_t Count() { return looks.size(); }

} // namespace npcs

void Walker::Update(float dt, bool present)
{
    time_ += dt;
    switch (phase_)
    {
    case Phase::Waiting:
        below_ = kStartBelow;
        if (time_ >= wait_)
        {
            phase_ = Phase::Coming;
            time_ = 0.0f;
        }
        break;
    case Phase::Coming:
        below_ = std::max(0.0f, below_ - kWalkSpeed * dt);
        if (below_ <= 0.0f)
            phase_ = Phase::Standing;
        if (!present)
        {
            phase_ = Phase::Turning;
            time_ = 0.0f;
        }
        break;
    case Phase::Standing:
        if (!present)
        {
            phase_ = Phase::Turning;
            time_ = 0.0f;
        }
        break;
    case Phase::Turning:
    {
        const Sheet& walk = looks[look_].walk;
        const auto [first, last] = walk.Part("turn");
        if (time_ >= walk.Length(first, last))
        {
            phase_ = Phase::Going;
            time_ = 0.0f;
        }
        break;
    }
    case Phase::Going:
        below_ += kWalkSpeed * dt;
        if (below_ > kStartBelow + 40)
            phase_ = Phase::Gone;
        break;
    case Phase::Gone:
        break;
    }
}

void Walker::Draw(sa::Backend& backend, sa::Vec2 spot) const
{
    if (look_ >= looks.size() || phase_ == Phase::Waiting || phase_ == Phase::Gone)
        return;
    const NpcLook& look = looks[look_];
    const Sheet* s = &look.idle;
    int frame = 0;
    if (phase_ == Phase::Coming)
    {
        s = &look.walk;
        const auto [first, last] = s->Part("away");
        frame = s->FrameAt(time_, first, last, true);
    }
    else if (phase_ == Phase::Turning)
    {
        s = &look.walk;
        const auto [first, last] = s->Part("turn");
        frame = s->FrameAt(time_, first, last, false);
    }
    else if (phase_ == Phase::Going)
    {
        s = &look.walk;
        const auto [first, last] = s->Part("toward");
        frame = s->FrameAt(time_, first, last, true);
    }
    // Match the protagonist and every other buyer at one fixed 200 px figure height.  Arrival depth
    // must not change character scale or make the planted idle pose jump in size.
    const float scale = stage::NpcHeight() / s->standing;
    DrawFrame(backend, *s, frame, {spot.x, spot.y + below_}, scale);
}
