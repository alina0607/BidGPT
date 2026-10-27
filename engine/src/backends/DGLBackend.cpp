#include "DGLBackend.h"

#include "../Image.h"

#include <DGL.h>

#include <cmath>
#include <iostream>
#include <string>
#include <utility>

namespace sa
{

namespace
{

// Characters typed since the last frame. Windows delivers them to the window procedure (WM_CHAR),
// which is a plain function, so they wait here until BeginFrame hands them to the backend.
std::string g_typing;

LRESULT CALLBACK WindowProc(HWND window, UINT message, WPARAM wParam, LPARAM lParam)
{
    if (message == WM_CHAR && wParam >= 32 && wParam < 127)    // printable ASCII: the game's font
        g_typing += static_cast<char>(wParam);
    int result = 0;
    if (DGL_System_HandleWindowsMessage(message, wParam, lParam, &result))
        return result;
    return DefWindowProc(window, message, wParam, lParam);
}

int ToVirtualKey(Key key)
{
    switch (key)
    {
    case Key::Escape: return VK_ESCAPE;
    case Key::Enter: return VK_RETURN;
    case Key::Space: return VK_SPACE;
    case Key::Backspace: return VK_BACK;
    case Key::C: return 'C';
    case Key::Left: return VK_LEFT;
    case Key::Right: return VK_RIGHT;
    case Key::Up: return VK_UP;
    case Key::Down: return VK_DOWN;
    case Key::W: return 'W';
    case Key::A: return 'A';
    case Key::S: return 'S';
    case Key::D: return 'D';
    case Key::Shift: return VK_SHIFT;
    case Key::J: return 'J';
    case Key::K: return 'K';
    }
    return 0;
}

// DGL draws in world space: origin at the window center, y up, 1 unit = 1 pixel at zoom 1.
Vec2 PixelToWorld(Vec2 pixel)
{
    const DGL_Vec2 screen{pixel.x, pixel.y};
    const DGL_Vec2 world = DGL_Camera_ScreenCoordToWorld(&screen);
    return {world.x, world.y};
}

} // namespace

bool DGLBackend::Init(const WindowDesc& desc)
{
    title_ = desc.title;

    DGL_SysInitInfo info{};
    info.mAppInstance = GetModuleHandle(nullptr);
    info.mShow = SW_SHOW;
    info.mWindowWidth = static_cast<unsigned>(desc.width);
    info.mWindowHeight = static_cast<unsigned>(desc.height);
    info.mMaxFrameRate = 60;
    info.mClassStyle = CS_HREDRAW | CS_VREDRAW;
    info.mWindowStyle = WS_OVERLAPPEDWINDOW;
    info.mWindowTitle = title_.c_str();
    info.mCreateConsole = FALSE;
    info.mWindowIcon = 0;
    info.pWindowsCallback = WindowProc;

    if (!DGL_System_Init(&info))
    {
        std::cerr << "DGL_System_Init failed: " << DGL_System_GetLastError() << '\n';
        return false;
    }
    initialized_ = true;

    // One white unit quad, scaled/rotated/tinted per draw, so nothing is allocated per frame.
    const DGL_Color white{1.0f, 1.0f, 1.0f, 1.0f};
    const DGL_Vec2 uv{0.0f, 0.0f};
    const DGL_Vec2 p0{-0.5f, -0.5f}, p1{0.5f, -0.5f}, p2{0.5f, 0.5f}, p3{-0.5f, 0.5f};
    DGL_Graphics_StartMesh();
    DGL_Graphics_AddTriangle(&p0, &white, &uv, &p1, &white, &uv, &p2, &white, &uv);
    DGL_Graphics_AddTriangle(&p0, &white, &uv, &p2, &white, &uv, &p3, &white, &uv);
    unitQuad_ = DGL_Graphics_EndMesh();

    DGL_Graphics_SetBlendMode(DGL_BM_BLEND);
    return unitQuad_ != nullptr;
}

void DGLBackend::Shutdown()
{
    if (!initialized_)
        return;
    DGL_Graphics_FreeMesh(&unitQuad_);
    for (auto& [size, mesh] : spriteMeshes_)
        DGL_Graphics_FreeMesh(&mesh);
    for (Texture& t : textures_)
        DGL_Graphics_FreeTexture(&t.texture);
    DGL_System_Exit();
    initialized_ = false;
}

bool DGLBackend::BeginFrame(Color clear)
{
    DGL_System_FrameControl();
    DGL_System_Update();                    // pumps window messages, so WM_CHAR has arrived
    if (!DGL_System_DoesWindowExist())
        return false;
    typed_ = std::move(g_typing);
    g_typing.clear();

    const DGL_Color background{clear.r, clear.g, clear.b, clear.a};
    DGL_Graphics_SetBackgroundColor(&background);
    DGL_Graphics_StartDrawing();
    DGL_Graphics_SetShaderMode(DGL_PSM_COLOR, DGL_VSM_DEFAULT);
    return true;
}

void DGLBackend::EndFrame() { DGL_Graphics_FinishDrawing(); }

Vec2 DGLBackend::MousePosition() const
{
    const DGL_Vec2 p = DGL_Input_GetMousePosition();
    return {p.x, p.y};
}

bool DGLBackend::MouseDown(MouseButton button) const
{
    return DGL_Input_KeyDown(button == MouseButton::Left ? VK_LBUTTON : VK_RBUTTON);
}

bool DGLBackend::KeyDown(Key key) const
{
    return DGL_Input_KeyDown(static_cast<unsigned char>(ToVirtualKey(key)));
}

bool DGLBackend::KeyPressed(Key key) const
{
    return DGL_Input_KeyTriggered(static_cast<unsigned char>(ToVirtualKey(key)));
}

void DGLBackend::DrawLine(Vec2 from, Vec2 to, float thickness, Color color)
{
    const Vec2 a = PixelToWorld(from);
    const Vec2 b = PixelToWorld(to);
    const Vec2 d = b - a;
    const float length = std::sqrt(d.x * d.x + d.y * d.y);
    // Extend by the thickness so consecutive segments overlap instead of leaving gaps.
    DrawQuad((a + b) * 0.5f, {length + thickness, thickness}, std::atan2(d.y, d.x), color);
}

void DGLBackend::DrawRect(Vec2 topLeft, Vec2 size, Color color)
{
    const Vec2 a = PixelToWorld(topLeft);
    const Vec2 b = PixelToWorld(topLeft + size);
    DrawQuad((a + b) * 0.5f, {std::fabs(b.x - a.x), std::fabs(b.y - a.y)}, 0.0f, color);
}

void DGLBackend::DrawQuad(Vec2 worldCenter, Vec2 worldSize, float radians, Color color)
{
    const DGL_Vec2 position{worldCenter.x, worldCenter.y};
    const DGL_Vec2 scale{worldSize.x, worldSize.y};
    // A tint with alpha 1 fully replaces the white mesh color.
    const DGL_Color tint{color.r, color.g, color.b, 1.0f};
    DGL_Graphics_SetCB_TransformData(&position, &scale, radians);
    DGL_Graphics_SetCB_TintColor(&tint);
    DGL_Graphics_SetCB_Alpha(color.a);
    DGL_Graphics_DrawMesh(unitQuad_, DGL_DM_TRIANGLELIST);
}

TextureId DGLBackend::LoadTexture(const std::string& path, TextureFilter filter)
{
    // Decoded here rather than by DGL_Graphics_LoadTexture, so the size is known (DGL doesn't say).
    const auto image = LoadImageRGBA(path);
    DGL_Texture* texture = image ? DGL_Graphics_LoadTextureFromMemory(image->rgba.data(), image->width, image->height)
                                 : nullptr;
    if (!texture)
    {
        std::cerr << "could not load texture " << path << '\n';
        return 0;
    }
    textures_.push_back({texture, {static_cast<float>(image->width), static_cast<float>(image->height)}, filter});
    return static_cast<TextureId>(textures_.size());
}

TextureId DGLBackend::CreateTexture(int width, int height, const unsigned char* rgba, TextureFilter filter)
{
    DGL_Texture* texture = width > 0 && height > 0 && rgba
                               ? DGL_Graphics_LoadTextureFromMemory(rgba, width, height)
                               : nullptr;
    if (!texture)
        return 0;
    textures_.push_back({texture, {static_cast<float>(width), static_cast<float>(height)}, filter});
    return static_cast<TextureId>(textures_.size());
}

void DGLBackend::UpdateTexture(TextureId texture, const unsigned char* rgba)
{
    if (texture < 1 || texture > textures_.size() || !rgba)
        return;
    // DGL has no way to change a texture's pixels: the old one is replaced by a new one of the same size.
    Texture& t = textures_[texture - 1];
    DGL_Texture* fresh =
        DGL_Graphics_LoadTextureFromMemory(rgba, static_cast<int>(t.size.x), static_cast<int>(t.size.y));
    if (!fresh)
        return;
    DGL_Graphics_FreeTexture(&t.texture);
    t.texture = fresh;
}

Vec2 DGLBackend::TextureSize(TextureId texture) const
{
    return texture >= 1 && texture <= textures_.size() ? textures_[texture - 1].size : Vec2{};
}

DGL_Mesh* DGLBackend::SpriteMesh(Vec2 uvSize)
{
    DGL_Mesh*& mesh = spriteMeshes_[{uvSize.x, uvSize.y}];
    if (!mesh)
    {
        // World y points up while texture v = 0 is the image's top row: the top edge gets v = 0.
        const DGL_Color white{1.0f, 1.0f, 1.0f, 1.0f};
        const DGL_Vec2 tl{-0.5f, 0.5f}, tr{0.5f, 0.5f}, br{0.5f, -0.5f}, bl{-0.5f, -0.5f};
        const DGL_Vec2 uvTL{0.0f, 0.0f}, uvTR{uvSize.x, 0.0f}, uvBR{uvSize.x, uvSize.y}, uvBL{0.0f, uvSize.y};
        DGL_Graphics_StartMesh();
        DGL_Graphics_AddTriangle(&tl, &white, &uvTL, &tr, &white, &uvTR, &br, &white, &uvBR);
        DGL_Graphics_AddTriangle(&tl, &white, &uvTL, &br, &white, &uvBR, &bl, &white, &uvBL);
        mesh = DGL_Graphics_EndMesh();
    }
    return mesh;
}

void DGLBackend::DrawSprite(TextureId texture, UVRect source, Vec2 topLeft, Vec2 size, float alpha)
{
    DrawTextured(texture, source, topLeft, size, {0.0f, 0.0f, 0.0f, 0.0f}, alpha);
}

void DGLBackend::DrawSpriteFilled(TextureId texture, UVRect source, Vec2 topLeft, Vec2 size, Color color)
{
    // DGL blends texel and tint by the tint's alpha, keeping the texel's alpha: tint alpha 1 paints the
    // shape in the color.
    DrawTextured(texture, source, topLeft, size, {color.r, color.g, color.b, 1.0f}, color.a);
}

void DGLBackend::DrawTextured(TextureId texture, UVRect source, Vec2 topLeft, Vec2 size, Color tint, float alpha)
{
    if (texture < 1 || texture > textures_.size())
        return;
    const Texture& t = textures_[texture - 1];
    const Vec2 a = PixelToWorld(topLeft);
    const Vec2 b = PixelToWorld(topLeft + size);
    const DGL_Vec2 position{(a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f};
    const DGL_Vec2 scale{std::fabs(b.x - a.x), std::fabs(b.y - a.y)};
    const DGL_Vec2 offset{source.offset.x, source.offset.y};
    DGL_Graphics_SetShaderMode(DGL_PSM_TEXTURE, DGL_VSM_DEFAULT);
    DGL_Graphics_SetTextureSamplerData(t.filter == TextureFilter::Smooth ? DGL_TSM_LINEAR : DGL_TSM_POINT,
                                       DGL_AM_CLAMP);
    DGL_Graphics_SetTexture(t.texture);
    DGL_Graphics_SetCB_TextureOffset(&offset);
    DGL_Graphics_SetCB_TransformData(&position, &scale, 0.0f);
    const DGL_Color dglTint{tint.r, tint.g, tint.b, tint.a};
    DGL_Graphics_SetCB_TintColor(&dglTint);
    DGL_Graphics_SetCB_Alpha(alpha);
    DGL_Graphics_DrawMesh(SpriteMesh(source.size), DGL_DM_TRIANGLELIST);
    // Shapes and text are drawn in color mode; leave it that way for them.
    const DGL_Vec2 noOffset{0.0f, 0.0f};
    DGL_Graphics_SetCB_TextureOffset(&noOffset);
    DGL_Graphics_SetShaderMode(DGL_PSM_COLOR, DGL_VSM_DEFAULT);
}

} // namespace sa
