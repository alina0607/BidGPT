#pragma once

#include "sa/Math.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace sa
{

enum class MouseButton
{
    Left,
    Right,
};

enum class Key
{
    Escape,
    Enter,
    Space,
    Backspace,
    C,
    Left,
    Right,
    Up,
    Down,
    W,
    A,
    S,
    D,
    Shift,
    J,
    K,
};

struct WindowDesc
{
    std::string title = "Sell Anything";
    int width = 1280;
    int height = 720;
};

// A loaded image; 0 means none.
using TextureId = unsigned;

// How a texture is sampled when drawn at another size: Pixel keeps pixel art crisp (nearest texel),
// Smooth blends neighbouring texels for painted art and font glyphs.
enum class TextureFilter
{
    Pixel,
    Smooth,
};

// Which part of a texture to draw, in 0..1 texture coordinates with v = 0 at the image's top row:
// the whole image is {{0, 0}, {1, 1}}; cell (c, r) of a cols × rows sprite sheet is
// {{c / cols, r / rows}, {1 / cols, 1 / rows}} (see sa/Sprite.h).
struct UVRect
{
    Vec2 offset{0.0f, 0.0f};
    Vec2 size{1.0f, 1.0f};
};

// Everything the engine needs from the platform: a window, input, and 2D drawing.
// Coordinates are window pixels with the origin at the top-left and y pointing down,
// the same convention as the Quick, Draw! stroke data.
class Backend
{
public:
    virtual ~Backend() = default;

    virtual std::string_view Name() const = 0;

    virtual bool Init(const WindowDesc& desc) = 0;
    virtual void Shutdown() = 0;

    // Pumps window events and starts a frame. Returns false once the window is closed.
    virtual bool BeginFrame(Color clear) = 0;
    virtual void EndFrame() = 0;

    virtual Vec2 MousePosition() const = 0;
    virtual bool MouseDown(MouseButton button) const = 0;
    // True only on the frame the key went down.
    virtual bool KeyPressed(Key key) const = 0;
    // True on every frame the key is held.
    virtual bool KeyDown(Key key) const
    {
        (void)key;
        return false;
    }
    // Printable ASCII characters typed since the last frame (for text fields).
    virtual std::string TypedText() const { return {}; }

    virtual void DrawLine(Vec2 from, Vec2 to, float thickness, Color color) = 0;
    virtual void DrawRect(Vec2 topLeft, Vec2 size, Color color) = 0;

    // Images for sprites. 0 if the file can't be read, or if this backend has no textures (then
    // DrawSprite draws nothing, so a game still runs without its art).
    virtual TextureId LoadTexture(const std::string& path, TextureFilter filter = TextureFilter::Pixel)
    {
        (void)path, (void)filter;
        return 0;
    }
    virtual Vec2 TextureSize(TextureId texture) const
    {
        (void)texture;
        return {};
    }
    // A texture made from 8-bit RGBA pixels, rows top to bottom: a picture that changes while the game runs
    // (a painting the AI is still painting). 0 if this backend has no textures.
    virtual TextureId CreateTexture(int width, int height, const unsigned char* rgba,
                                    TextureFilter filter = TextureFilter::Smooth)
    {
        (void)width, (void)height, (void)rgba, (void)filter;
        return 0;
    }
    // Replaces every pixel of a texture made by CreateTexture (same size and layout).
    virtual void UpdateTexture(TextureId texture, const unsigned char* rgba) { (void)texture, (void)rgba; }
    // Part of a texture stretched over a rectangle, sampled as it was loaded; alpha fades it.
    virtual void DrawSprite(TextureId texture, UVRect source, Vec2 topLeft, Vec2 size, float alpha = 1.0f)
    {
        (void)texture, (void)source, (void)topLeft, (void)size, (void)alpha;
    }
    // The shape of part of a white texture (its alpha) in one color, for font glyphs.
    virtual void DrawSpriteFilled(TextureId texture, UVRect source, Vec2 topLeft, Vec2 size, Color color)
    {
        (void)texture, (void)source, (void)topLeft, (void)size, (void)color;
    }

};

// Names of the backends compiled into this build, preferred one first.
std::vector<std::string_view> AvailableBackends();

// Creates a backend by name; an empty name picks the preferred one.
// Returns nullptr if the name is not compiled into this build.
std::unique_ptr<Backend> CreateBackend(std::string_view name = {});

} // namespace sa
