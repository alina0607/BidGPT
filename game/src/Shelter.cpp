#include "Shelter.h"

#include "Stage.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <iostream>

namespace
{

constexpr float kPollEvery = 1.0f / 20.0f;   // s: the canvas changes every ~20 ms; 20 looks a second are plenty

// Standard base64 (no line breaks) into bytes; false on a character outside the alphabet.
bool DecodeBase64(const std::string& text, std::vector<unsigned char>& out)
{
    static const auto table = [] {
        std::array<int, 256> t{};
        t.fill(-1);
        const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i)
            t[static_cast<unsigned char>(alphabet[i])] = i;
        return t;
    }();
    out.clear();
    out.reserve(text.size() / 4 * 3);
    unsigned buffer = 0;
    int bits = 0;
    for (char c : text)
    {
        if (c == '=')
            break;
        const int v = table[static_cast<unsigned char>(c)];
        if (v < 0)
            return false;
        buffer = (buffer << 6) | static_cast<unsigned>(v);
        bits += 6;
        if (bits >= 8)
        {
            bits -= 8;
            out.push_back(static_cast<unsigned char>((buffer >> bits) & 0xFF));
        }
    }
    return true;
}

} // namespace

void Shelter::Start(sa::Backend& backend)
{
    (void)backend;
    stage::ClearEasels();
    eyeShown_.fill(false);
    recent_.clear();
    wantPaused_ = paused_ = false;  // a new run starts unpaused
    cat_ = 0;
    have_ = -1;
    strokes_ = 0;
    error_.clear();
    started_ = std::chrono::steady_clock::now();
    state_ = State::Starting;
    if (client_.Busy())
    {
        restart_ = true;            // a poll of the old run is out; its reply is dropped
        return;
    }
    client_.Send({{"type", "shelter_paint"}});
}

void Shelter::Update(sa::Backend& backend, float dt)
{
    if (state_ == State::Idle)
        return;
    sincePoll_ += dt;
    if (const auto reply = client_.Poll())
    {
        if (restart_)
        {
            restart_ = false;
            client_.Send({{"type", "shelter_paint"}});
            return;
        }
        Take(backend, *reply);
    }
    if (state_ == State::Painting && !client_.Busy() && wantPaused_ != paused_)
    {
        client_.Send({{"type", "shelter_pause"}, {"paused", wantPaused_}, {"cat", cat_}});
        have_ = -1;                 // the reply carries the canvas again
    }
    else if (state_ == State::Painting && !client_.Busy() && sincePoll_ >= kPollEvery)
        Poll(backend);
}

void Shelter::Poll(sa::Backend& backend)
{
    (void)backend;
    sincePoll_ = 0.0f;
    client_.Send({{"type", "shelter_poll"}, {"cat", cat_}, {"have", have_}});
}

void Shelter::Take(sa::Backend& backend, const nlohmann::json& reply)
{
    if (reply.value("type", "") != "shelter_state")
    {
        error_ = reply.value("message", "The painting server did not answer.");
        state_ = State::Failed;
        std::cerr << "shelter: " << error_ << '\n';
        return;
    }
    if (state_ == State::Starting)
    {
        run_ = reply.value("run", 0);
        cats_ = std::min(reply.value("cats", 8), stage::EaselCount());
        state_ = State::Painting;
    }
    if (reply.value("run", -1) != run_ || reply.value("cat", -1) != cat_)
        return;                     // an answer about an older run, or another cat
    if (reply.contains("error"))
    {
        error_ = reply["error"].get<std::string>();
        state_ = State::Failed;
        return;
    }
    strokes_ = reply.value("strokes", 0);
    total_ = reply.value("total", 0);
    eyeStrokes_ = reply.value("eye_strokes", 0);
    phase_ = reply.value("phase", std::string());
    layer_ = reply.value("layer", std::string());
    layerIndex_ = reply.value("layer_index", 0);
    layers_ = reply.value("layers", 0);
    paused_ = reply.value("paused", false);
    catSeconds_ = reply.contains("cat_seconds") && reply["cat_seconds"].is_number() ? reply["cat_seconds"].get<double>() : 0.0;
    recent_.clear();
    if (reply.contains("recent") && reply["recent"].is_array())
        for (const auto& p : reply["recent"])
            if (p.is_array() && p.size() == 2)
                recent_.push_back({p[0].get<float>(), p[1].get<float>()});
    const int width = reply.value("width", 0), height = reply.value("height", 0);
    if (reply.contains("image") && width > 0 && height > 0 &&
        DecodeBase64(reply["image"].get_ref<const std::string&>(), rgba_) &&
        rgba_.size() == static_cast<size_t>(width) * height * 3)
    {
        // RGB from the server; the textures are RGBA.
        std::vector<unsigned char> pixels(static_cast<size_t>(width) * height * 4, 255);
        for (size_t i = 0, n = static_cast<size_t>(width) * height; i < n; ++i)
            for (int c = 0; c < 3; ++c)
                pixels[i * 4 + c] = rgba_[i * 3 + c];
        stage::PaintEasel(backend, cat_, width, height, pixels.data());
        have_ = reply.value("version", -1);
        // The eye canvas (as wide and high as the face canvas), once the eyes are being painted.
        if (reply.contains("eye_image") && reply.contains("eye_box") && reply["eye_box"].size() == 4 &&
            DecodeBase64(reply["eye_image"].get_ref<const std::string&>(), eyeRgb_) &&
            eyeRgb_.size() == static_cast<size_t>(width) * height * 3)
        {
            for (size_t i = 0, n = static_cast<size_t>(width) * height; i < n; ++i)
                for (int c = 0; c < 3; ++c)
                    pixels[i * 4 + c] = eyeRgb_[i * 3 + c];
            sa::TextureId& tex = eyeCanvas_[cat_];
            const sa::Vec2 size = backend.TextureSize(tex);
            if (tex && size.x == width && size.y == height)
                backend.UpdateTexture(tex, pixels.data());
            else
                tex = backend.CreateTexture(width, height, pixels.data(), sa::TextureFilter::Smooth);
            eyeShown_[cat_] = tex != 0;
            for (int k = 0; k < 4; ++k)
                eyeBox_[cat_][k] = reply["eye_box"][k].get<float>();
        }
    }
    // A finished cat moves on once its last canvas is on the easel.
    if (reply.value("finished", false) && have_ == reply.value("version", -2))
    {
        std::cout << "shelter: cat " << cat_ + 1 << " painted (" << strokes_ << " strokes)\n";
        ++cat_;
        have_ = -1;
        strokes_ = 0;
        if (cat_ >= cats_)
        {
            seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - started_).count();
            state_ = State::Done;
            std::cout << "shelter: " << cats_ << " cats painted in " << seconds_ << " s (game clock, from the press)"
                      << std::endl;
        }
    }
}

std::string Shelter::CatStatus(int i) const
{
    char line[96];
    if (state_ == State::Painting && i == cat_)
        return Status();
    if (state_ == State::Done || (state_ == State::Painting && i < cat_))
        std::snprintf(line, sizeof line, "Cat %d of %d: finished", i + 1, cats_);
    else
        std::snprintf(line, sizeof line, "Cat %d of %d: waiting its turn", i + 1, cats_);
    return line;
}

std::string Shelter::Status() const
{
    char line[160];
    switch (state_)
    {
    case State::Idle:
        return "";
    case State::Starting:
        return "The AI is getting its brushes ready...";
    case State::Painting:
    {
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - started_).count();
        if (total_ > 0)
            std::snprintf(line, sizeof line, "The AI is painting cat %d of %d: %d of %d strokes  (%.0f s)", cat_ + 1,
                          cats_, strokes_, total_, s);
        else if (phase_ == "eyes")
            std::snprintf(line, sizeof line, "%sCat %d of %d at %.1f s, the eyes: %s (layer %d of %d), %d strokes",
                          paused_ ? "PAUSED  " : "", cat_ + 1, cats_, catSeconds_, layer_.c_str(), layerIndex_, layers_,
                          eyeStrokes_);
        else if (!layer_.empty())
            std::snprintf(line, sizeof line, "%sCat %d of %d at %.1f s, the face: %s (layer %d of %d), %d strokes",
                          paused_ ? "PAUSED  " : "", cat_ + 1, cats_, catSeconds_, layer_.c_str(), layerIndex_, layers_,
                          strokes_);
        else
            std::snprintf(line, sizeof line, "The AI is painting cat %d of %d: %d strokes  (%.0f s)", cat_ + 1, cats_,
                          strokes_, s);
        return line;
    }
    case State::Done:
        std::snprintf(line, sizeof line, "The AI painted %d cats in %.0f s. Reset paints %d new ones.", cats_,
                      seconds_, cats_);
        return line;
    case State::Failed:
        return "The painting stopped: " + error_;
    }
    return "";
}
