#pragma once

#include "sa/Backend.h"

#include "../Image.h"

#include <chrono>
#include <cstdlib>
#include <thread>

namespace sa
{

// Headless backend: no window, draws nothing, closes itself after a fixed number of frames
// (120, or SA_EXIT_AFTER=N like the OpenGL backend). A frame takes about a millisecond, so
// frame-counted game logic (autoplay waits, timeouts) behaves the same as with a window.
// Used for CI smoke and end-to-end tests and for running the game logic without a GPU.
class NullBackend final : public Backend
{
public:
    static constexpr std::string_view kName = "null";

    explicit NullBackend(int maxFrames = 120) : maxFrames_(maxFrames)
    {
        if (const char* exitAfter = std::getenv("SA_EXIT_AFTER"))
            maxFrames_ = std::atoi(exitAfter);
    }

    std::string_view Name() const override { return kName; }

    bool Init(const WindowDesc&) override { return true; }
    void Shutdown() override {}

    bool BeginFrame(Color) override { return frame_++ < maxFrames_; }
    void EndFrame() override { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }

    Vec2 MousePosition() const override { return {}; }
    bool MouseDown(MouseButton) const override { return false; }
    bool KeyPressed(Key) const override { return false; }

    void DrawLine(Vec2, Vec2, float, Color) override {}
    void DrawRect(Vec2, Vec2, Color) override {}

    // Images are really read, so a headless run still fails on missing or broken art.
    TextureId LoadTexture(const std::string& path, TextureFilter) override
    {
        const auto image = LoadImageRGBA(path);
        if (!image)
            return 0;
        sizes_.push_back({static_cast<float>(image->width), static_cast<float>(image->height)});
        return static_cast<TextureId>(sizes_.size());
    }
    TextureId CreateTexture(int width, int height, const unsigned char* rgba, TextureFilter) override
    {
        if (width <= 0 || height <= 0 || !rgba)
            return 0;
        sizes_.push_back({static_cast<float>(width), static_cast<float>(height)});
        return static_cast<TextureId>(sizes_.size());
    }
    Vec2 TextureSize(TextureId texture) const override
    {
        return texture >= 1 && texture <= sizes_.size() ? sizes_[texture - 1] : Vec2{};
    }

private:
    std::vector<Vec2> sizes_;
    int maxFrames_;
    int frame_ = 0;
};

} // namespace sa
