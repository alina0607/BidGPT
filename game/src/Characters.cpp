#include "Characters.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace
{

// The sprite state each server mood plays (docs/protocol.md → docs/art.md).
std::string StateFor(const std::string& mood)
{
    if (mood == "left")
        return "leave";
    if (mood == "won")
        return "win";
    if (mood == "talk" || mood == "happy" || mood == "unhappy")
        return mood;
    return "idle";
}

// Looks are picked by a stable hash of the name (std::hash may differ between standard libraries).
size_t NameHash(const std::string& name)
{
    size_t h = 2166136261u;
    for (unsigned char c : name)
        h = (h ^ c) * 16777619u;
    return h;
}

} // namespace

std::string AssetDir()
{
    if (const char* dir = std::getenv("SA_ASSETS"))
        return dir;
    return SA_SOURCE_ASSETS;
}

void CharacterArt::Load(sa::Backend& backend, const std::string& dir)
{
    loaded_ = true;
    namespace fs = std::filesystem;
    std::error_code error;
    std::vector<fs::path> folders;
    for (const auto& entry : fs::directory_iterator(dir, error))
        if (entry.is_directory())
            folders.push_back(entry.path());
    std::sort(folders.begin(), folders.end());   // directory order differs between file systems
    for (const fs::path& folder : folders)
    {
        const fs::path meta = folder / (folder.filename().string() + ".json");
        std::ifstream file(meta);
        if (!file)
            continue;
        const auto json = nlohmann::json::parse(file, nullptr, false);
        if (json.is_discarded() || !json.contains("states"))
        {
            std::cerr << "skipping look " << folder.filename() << ": unreadable " << meta.filename() << '\n';
            continue;
        }
        Look look{folder.filename().string(), {}};
        bool complete = true;
        for (const auto& [state, s] : json["states"].items())
        {
            const sa::TextureId texture = backend.LoadTexture((folder / s["file"].get<std::string>()).string());
            complete = complete && texture != 0;
            look.states[state] = {{texture, s["cols"], s["rows"], s["frames"]}, s["frame_time"], s["loop"]};
        }
        if (complete)
            looks_.push_back(std::move(look));
        else
            std::cerr << "skipping look " << folder.filename() << ": a sheet failed to load\n";
    }
}

const Look* CharacterArt::LookFor(const std::string& buyerName) const
{
    return looks_.empty() ? nullptr : &looks_[NameHash(buyerName) % looks_.size()];
}

void BuyerSprite::SetMood(const std::string& mood, const Look& look)
{
    if (mood == mood_)
        return;
    mood_ = mood;
    Play(StateFor(mood), look);
}

void BuyerSprite::Update(float dt, const Look& look)
{
    animation_.Update(dt);
    if (animation_.Done() && (state_ == "happy" || state_ == "unhappy"))
        Play("idle", look);
}

void BuyerSprite::Play(const std::string& state, const Look& look)
{
    state_ = state;
    const auto it = look.states.find(state);
    if (it != look.states.end())
        animation_.Play(it->second.sheet.frames, it->second.frameTime, it->second.loop);
}

void BuyerSprite::Draw(sa::Backend& backend, const Look& look, sa::Vec2 topLeft, sa::Vec2 size) const
{
    const auto it = look.states.find(state_);
    if (it == look.states.end() || !Visible())
        return;
    const int frame = animation_.Frame();
    // Walking off also fades, so the empty card reads as "gone" rather than "missing art".
    const float alpha = state_ == "leave" ? 1.0f - 0.8f * static_cast<float>(frame) / it->second.sheet.frames : 1.0f;
    it->second.sheet.Draw(backend, frame, topLeft, size, alpha);
}
