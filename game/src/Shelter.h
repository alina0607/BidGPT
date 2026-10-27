#pragma once

#include "MlClient.h"

#include "sa/Backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <string>
#include <utility>
#include <vector>

// The cat shelter's eight easels, painted live by Brush GPT on the ML server (docs/protocol.md, shelter_*):
// one cat after another, left easel first, each canvas shown stroke by stroke while the model paints it.
// It has its own connection to the server, so asking for canvases never waits behind a drawing guess or a
// buyer's reply.
class Shelter
{
public:
    Shelter(std::string host, int port) : client_(std::move(host), port) {}

    // Blanks the easels and asks for eight new cats (the first visit, and Reset).
    void Start(sa::Backend& backend);
    // Asks for the canvas being painted and puts what arrives on its easel. Call every frame.
    void Update(sa::Backend& backend, float dt);

    bool Started() const { return state_ != State::Idle; }
    bool Starting() const { return state_ == State::Starting; }
    bool Done() const { return state_ == State::Done; }
    // One line for the status panel: which cat is being painted, or how long the eight took.
    std::string Status() const;
    // One line about cat i (0..7): being painted (as Status), finished, or still waiting its turn.
    std::string CatStatus(int i) const;
    // The cat being painted (-1: none), whether its eyes are being painted, and where its last strokes went
    // (canvas px, 0..256: on the face, or on the eye canvas while the eyes are painted; oldest first).
    int PaintingCat() const { return state_ == State::Painting ? cat_ : -1; }
    // Holds the painting (and the cat's clock) on the server, or lets it go on; Paused is what the server says.
    void SetPaused(bool on) { wantPaused_ = on; }
    bool WantPaused() const { return wantPaused_; }
    bool Paused() const { return paused_; }
    bool CanPause() const { return state_ == State::Painting; }
    bool PaintingEyes() const { return state_ == State::Painting && phase_ == "eyes"; }
    const std::vector<sa::Vec2>& Recent() const { return recent_; }
    // Cat i's eye canvas (the eye model's own zoomed view, 0 until its eyes are painted) and where it is on the
    // face canvas (px: x0, y0, x1, y1).
    sa::TextureId EyeCanvas(int i) const { return i >= 0 && i < static_cast<int>(eyeCanvas_.size()) && eyeShown_[i] ? eyeCanvas_[i] : 0; }
    std::array<float, 4> EyeBox(int i) const { return eyeBox_[std::clamp(i, 0, 7)]; }

private:
    enum class State
    {
        Idle,
        Starting,       // shelter_paint is out
        Painting,
        Done,
        Failed,
    };

    void Poll(sa::Backend& backend);
    void Take(sa::Backend& backend, const nlohmann::json& reply);

    MlClient client_;
    State state_ = State::Idle;
    bool restart_ = false;          // Reset pressed while a reply was still out: start when it is in
    int run_ = 0;                   // the server's run the easels show
    int cat_ = 0;                   // the cat being fetched (0..7)
    int have_ = -1;                 // the version of that cat's canvas already on its easel
    int strokes_ = 0, total_ = 0;   // of the cat being fetched (total 0: the models choose how many)
    int eyeStrokes_ = 0;            // of those, the eye model's
    std::string phase_;             // "face" (L8) or "eyes" (the eye model) for the cat being fetched
    std::string layer_;             // the layer being painted, and its place among the layers
    int layerIndex_ = 0, layers_ = 0;
    std::vector<sa::Vec2> recent_;
    bool wantPaused_ = false, paused_ = false;
    double catSeconds_ = 0.0;       // the cat being fetched: seconds painted, pauses left out (the server's clock)
    std::array<sa::TextureId, 8> eyeCanvas_{};
    std::array<bool, 8> eyeShown_{};
    std::array<std::array<float, 4>, 8> eyeBox_{};
    std::vector<unsigned char> eyeRgb_;
    int cats_ = 8;
    float sincePoll_ = 0.0f;
    std::vector<unsigned char> rgba_;
    std::chrono::steady_clock::time_point started_;
    double seconds_ = 0.0;          // the game's own clock, from the press to the eighth cat's last stroke
    std::string error_;
};
