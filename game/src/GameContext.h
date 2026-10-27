#pragma once

#include "Characters.h"
#include "DoodleBoard.h"
#include "MlClient.h"
#include "Girl.h"
#include "Shelter.h"

#include "sa/Math.h"

#include <string>
#include <utility>
#include <vector>

using Stroke = std::vector<sa::Vec2>;

struct Candidate
{
    std::string label;
    float prob = 0.0f;
};

// State shared by the scenes of one game session.
struct GameContext
{
    GameContext(const std::string& host, int port) : client(host, port), shelter(host, port), doodle(host, port) {}

    MlClient client;
    Shelter shelter;                // the cat shelter's easels, painted by Brush GPT
    DoodleBoard doodle;             // the colour cat doodle board left of the stall, read live by the models
    bool autoplay = false;          // play by itself (screenshots, demos, the end-to-end test)
    bool sold = false;              // did a round end in a sale? (autoplay's exit code)
    std::vector<Stroke> drawing;    // the player's last drawing
    std::vector<Candidate> candidates;
    std::string session;            // id of the round being played
    CharacterArt art;               // buyer sprites, loaded on the first round
    Girl girl;                      // the player, walking the market on every screen
    int rounds = 0;

    const std::string& NewSession()
    {
        session = "round-" + std::to_string(++rounds);
        return session;
    }
};
