#pragma once

#include "sa/Backend.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

struct DGL_Mesh;
struct DGL_Texture;

namespace sa
{

// DigiPen Graphics Library (Direct3D 11, Windows only).
// DGL is not redistributable, so it is never committed; see docs/backends.md.
class DGLBackend final : public Backend
{
public:
    static constexpr std::string_view kName = "dgl";

    std::string_view Name() const override { return kName; }

    bool Init(const WindowDesc& desc) override;
    void Shutdown() override;

    bool BeginFrame(Color clear) override;
    void EndFrame() override;

    Vec2 MousePosition() const override;
    bool MouseDown(MouseButton button) const override;
    bool KeyPressed(Key key) const override;
    bool KeyDown(Key key) const override;
    std::string TypedText() const override { return typed_; }

    void DrawLine(Vec2 from, Vec2 to, float thickness, Color color) override;
    void DrawRect(Vec2 topLeft, Vec2 size, Color color) override;

    TextureId LoadTexture(const std::string& path, TextureFilter filter) override;
    TextureId CreateTexture(int width, int height, const unsigned char* rgba, TextureFilter filter) override;
    void UpdateTexture(TextureId texture, const unsigned char* rgba) override;
    Vec2 TextureSize(TextureId texture) const override;
    void DrawSprite(TextureId texture, UVRect source, Vec2 topLeft, Vec2 size, float alpha) override;
    void DrawSpriteFilled(TextureId texture, UVRect source, Vec2 topLeft, Vec2 size, Color color) override;
    // DGL blends texel and tint by the tint's alpha; {0,0,0,0} leaves the texture as it is.
    void DrawTextured(TextureId texture, UVRect source, Vec2 topLeft, Vec2 size, Color tint, float alpha);

private:
    void DrawQuad(Vec2 worldCenter, Vec2 worldSize, float radians, Color color);
    DGL_Mesh* SpriteMesh(Vec2 uvSize);

    struct Texture
    {
        DGL_Texture* texture = nullptr;
        Vec2 size;
        TextureFilter filter = TextureFilter::Pixel;
    };
    std::vector<Texture> textures_;                           // TextureId n is textures_[n - 1]
    // DGL offsets texture coordinates per draw but not their extent, so there is one unit quad per
    // cell size (like CS529's mesh3x3), made on first use.
    std::map<std::pair<float, float>, DGL_Mesh*> spriteMeshes_;

    std::string title_; // DGL keeps the pointer, so it must outlive the window.
    std::string typed_; // characters typed during the last frame
    DGL_Mesh* unitQuad_ = nullptr;
    bool initialized_ = false;
};

} // namespace sa
