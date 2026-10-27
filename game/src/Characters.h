#pragma once

#include "sa/Backend.h"
#include "sa/Sprite.h"

#include <map>
#include <string>
#include <vector>

// Buyer sprites (docs/art.md): every look is a folder with <look>.json and one sheet per state.
// A buyer always gets the same look (picked from their name), and plays the state that matches the
// mood the server reports for them.
struct StateSheet
{
    sa::SpriteSheet sheet;
    float frameTime = 0.1f;
    bool loop = true;
};

struct Look
{
    std::string name;
    std::map<std::string, StateSheet> states;   // idle, talk, happy, unhappy, leave, win
};

class CharacterArt
{
public:
    // Loads every look under `dir`; looks with a missing or unreadable sheet are skipped (with a
    // message), so the game still runs, without sprites, when the art isn't there.
    void Load(sa::Backend& backend, const std::string& dir);
    bool Loaded() const { return loaded_; }
    const Look* LookFor(const std::string& buyerName) const;
    size_t LookCount() const { return looks_.size(); }

private:
    bool loaded_ = false;
    std::vector<Look> looks_;
};

// One buyer's animation: follows their mood. A reaction (happy, unhappy) plays once and settles
// back to idle; leaving walks off and fades out; winning holds its last frame.
class BuyerSprite
{
public:
    void SetMood(const std::string& mood, const Look& look);
    void Update(float dt, const Look& look);
    void Draw(sa::Backend& backend, const Look& look, sa::Vec2 topLeft, sa::Vec2 size) const;

    const std::string& State() const { return state_; }
    int Frame() const { return animation_.Frame(); }
    // False once a buyer who left has finished walking off.
    bool Visible() const { return !(state_ == "leave" && animation_.Done()); }

private:
    void Play(const std::string& state, const Look& look);

    std::string mood_;
    std::string state_;
    sa::Animation animation_;
};

// Where the art lives: $SA_ASSETS, else the source tree this game was built from.
std::string AssetDir();
