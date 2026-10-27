#include "Stage.h"

#include "Characters.h"
#include "Sheets.h"
#include "Ui.h"

#include "sa/Text.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <utility>

namespace
{

constexpr float kScreenW = 1280.0f;
constexpr float kCameraEase = 6.0f;     // 1/s: how fast the camera catches up
constexpr float kFreeLeft = 300.0f;     // she walks between these screen x without the camera moving,
constexpr float kFreeRight = 980.0f;    // so the stall stays put while she wanders in front of it
constexpr float kBrush = 6.0f;          // canvas units, as on the old full-screen canvas
constexpr float kCorner = 16.0f;        // screen radius of rounded panels
constexpr float kSlice = 20.0f / 64.0f; // the corner's share of the 64-px rounded textures
constexpr float kShelterCatScale = 1.38f; // fill more of each easel while keeping its painting centred

struct Object
{
    sa::TextureId texture = 0;
    sa::Vec2 size;          // drawn size
    float x = 0.0f;         // world x of the centre
    float bottom = 0.0f;
};

struct Art
{
    bool tried = false;
    sa::TextureId background = 0;
    sa::Vec2 backgroundSize;
    sa::TextureId fill = 0, edge = 0;
    sa::TextureId uiPanel = 0, uiTitle = 0, uiButton = 0, uiButtonGreen = 0, uiButtonCream = 0,
                  uiSpeech = 0;
    std::vector<Object> objects;
    sa::Vec2 clothPos, clothSize;   // world
    Object stall;                   // the object whose meta has the cloth (drawn again by DrawStall)
    // The shelter's easels, left to right. Blank until Brush GPT paints them (Shelter.h): each painting is
    // a texture of its own, made the first time a canvas arrives and updated stroke by stroke after that.
    struct Easel
    {
        Object canvas;              // where the painting hangs
        sa::TextureId painting = 0;
        bool shown = false;
    };
    std::vector<Easel> easels;
    float shelterX = 0.0f;          // world x of the shelter's centre
    float ground = 545.0f;
    float girlStart = 300.0f;
    float npcHeight = 190.0f;
    float girlHeight = 200.0f;
    float buyerScale = 2.25f;
};

Art art;
float cameraX = -kScreenW / 2;   // world x at the screen's left edge: the stall in the middle
bool lookAtShelter = false;      // the camera shows the shelter instead of following the girl
bool lookAtDoodle = false;       // the camera shows the doodle station instead of following the girl

nlohmann::json ReadJson(const std::string& path)
{
    std::ifstream in(path);
    return in ? nlohmann::json::parse(in, nullptr, false) : nlohmann::json();
}

sa::Vec2 ClothScreenPos() { return stage::ToScreen(art.clothPos); }

} // namespace

namespace stage
{

void Load(sa::Backend& backend)
{
    if (art.tried)
        return;
    art.tried = true;
    const std::string dir = AssetDir() + "/stage";
    sa::LoadFont(backend, AssetDir() + "/fonts/nunito.png", AssetDir() + "/fonts/nunito.txt");
    sa::LoadFallbackFont(backend, AssetDir() + "/fonts/cjk.png", AssetDir() + "/fonts/cjk.txt");
    art.background = backend.LoadTexture(dir + "/background.png", sa::TextureFilter::Smooth);
    art.backgroundSize = backend.TextureSize(art.background);
    art.fill = backend.LoadTexture(dir + "/round_fill.png", sa::TextureFilter::Smooth);
    art.edge = backend.LoadTexture(dir + "/round_edge.png", sa::TextureFilter::Smooth);
    const std::string ui = AssetDir() + "/ui";
    art.uiPanel = backend.LoadTexture(ui + "/panel.png", sa::TextureFilter::Smooth);
    art.uiTitle = backend.LoadTexture(ui + "/title.png", sa::TextureFilter::Smooth);
    art.uiButton = backend.LoadTexture(ui + "/button.png", sa::TextureFilter::Smooth);
    art.uiButtonGreen = backend.LoadTexture(ui + "/button_green.png", sa::TextureFilter::Smooth);
    art.uiButtonCream = backend.LoadTexture(ui + "/button_cream.png", sa::TextureFilter::Smooth);
    art.uiSpeech = backend.LoadTexture(ui + "/speech.png", sa::TextureFilter::Smooth);
    const auto layout = ReadJson(dir + "/layout.json");
    if (layout.is_object())
    {
        art.ground = layout.value("ground", art.ground);
        art.girlStart = layout.value("girl_start", art.girlStart);
        art.npcHeight = layout.value("npc_height", art.npcHeight);
        art.girlHeight = layout.value("girl_height", art.girlHeight);
        art.buyerScale = layout.value("buyer_scale", art.buyerScale);
        for (const auto& o : layout.value("objects", nlohmann::json::array()))
        {
            const std::string image = o.value("image", "");
            Object object;
            object.texture = backend.LoadTexture(dir + "/" + image + ".png", sa::TextureFilter::Smooth);
            const sa::Vec2 tex = backend.TextureSize(object.texture);
            const float height = o.value("height", tex.y);
            const float k = tex.y > 0 ? height / tex.y : 0.0f;
            object.size = {tex.x * k, height};
            object.x = o.value("x", 0.0f);
            object.bottom = o.value("bottom", art.ground);
            art.objects.push_back(object);
            const auto meta = ReadJson(dir + "/" + image + ".json");
            if (meta.is_object() && meta.contains("easels"))
            {
                art.shelterX = object.x;
                for (const auto& e : meta["easels"])
                {
                    const sa::Vec2 topLeft{object.x - object.size.x / 2, object.bottom - object.size.y};
                    Object canvas;
                    const float x0 = e[0].get<float>(), y0 = e[1].get<float>();
                    const float x1 = e[2].get<float>(), y1 = e[3].get<float>();
                    canvas.size = {(x1 - x0) * k * kShelterCatScale, (y1 - y0) * k * kShelterCatScale};
                    canvas.x = topLeft.x + (x0 + x1) * 0.5f * k;
                    canvas.bottom = topLeft.y + (y0 + y1) * 0.5f * k + canvas.size.y / 2;
                    art.easels.push_back({canvas});
                }
            }
            if (meta.is_object() && meta.contains("cloth"))
            {
                const auto& c = meta["cloth"];
                const sa::Vec2 topLeft{object.x - object.size.x / 2, object.bottom - object.size.y};
                art.clothPos = {topLeft.x + c[0].get<float>() * k, topLeft.y + c[1].get<float>() * k};
                art.clothSize = {(c[2].get<float>() - c[0].get<float>()) * k, (c[3].get<float>() - c[1].get<float>()) * k};
                art.stall = object;
            }
        }
    }
}

void Follow(float x, float dt)
{
    const float onScreen = x - cameraX;
    const float target = lookAtShelter          ? art.shelterX - kScreenW / 2
                         : lookAtDoodle           ? kDoodleX - kScreenW / 2
                         : onScreen > kFreeRight  ? x - kFreeRight
                         : onScreen < kFreeLeft ? x - kFreeLeft
                                                : cameraX;
    cameraX += (target - cameraX) * std::min(1.0f, kCameraEase * dt);
}

sa::Vec2 ToScreen(sa::Vec2 world) { return {world.x - cameraX, world.y}; }

void DrawWorld(sa::Backend& backend)
{
    Load(backend);
    if (art.background && art.backgroundSize.x > 0)
    {
        // The strip repeats without a seam: as many copies as cover the screen.
        const float w = art.backgroundSize.x;
        for (float x = std::floor(cameraX / w) * w; x < cameraX + kScreenW; x += w)
            backend.DrawSprite(art.background, {}, {x - cameraX, 0}, art.backgroundSize);
    }
    else
        backend.DrawRect({0, 0}, {10000, 10000}, ui::kBackground);
    for (const Object& o : art.objects)
        backend.DrawSprite(o.texture, {}, ToScreen({o.x - o.size.x / 2, o.bottom - o.size.y}), o.size);
    for (int i = 0; i < EaselCount(); ++i)
    {
        if (!art.easels[i].shown)
            continue;
        sa::Vec2 pos, size;
        EaselScreenRect(i, pos, size);
        backend.DrawSprite(art.easels[i].painting, {}, pos, size);
    }
}

void EaselScreenRect(int i, sa::Vec2& topLeft, sa::Vec2& size)
{
    constexpr float kMat = 3.0f;   // a little of the easel's blank canvas shows round each painting
    const Object& c = art.easels[i].canvas;
    topLeft = ToScreen({c.x - c.size.x / 2 + kMat, c.bottom - c.size.y + kMat});
    size = {c.size.x - 2 * kMat, c.size.y - 2 * kMat};
}

sa::TextureId EaselPainting(int i)
{
    return i >= 0 && i < EaselCount() && art.easels[i].shown ? art.easels[i].painting : 0;
}

int EaselAt(sa::Vec2 screen)
{
    for (int i = 0; i < EaselCount(); ++i)
    {
        sa::Vec2 pos, size;
        EaselScreenRect(i, pos, size);
        if (EaselPainting(i) && screen.x >= pos.x && screen.x < pos.x + size.x && screen.y >= pos.y &&
            screen.y < pos.y + size.y)
            return i;
    }
    return -1;
}

int EaselCount() { return static_cast<int>(art.easels.size()); }

void ClearEasels()
{
    for (auto& easel : art.easels)
        easel.shown = false;
}

void PaintEasel(sa::Backend& backend, int i, int width, int height, const unsigned char* rgba)
{
    if (i < 0 || i >= EaselCount())
        return;
    Art::Easel& easel = art.easels[i];
    const sa::Vec2 size = backend.TextureSize(easel.painting);
    if (easel.painting && size.x == width && size.y == height)
        backend.UpdateTexture(easel.painting, rgba);
    else
        easel.painting = backend.CreateTexture(width, height, rgba, sa::TextureFilter::Smooth);
    easel.shown = easel.painting != 0;
}

void LookAtShelter(bool on)
{
    lookAtShelter = on && art.shelterX != 0.0f;
    if (lookAtShelter)
        lookAtDoodle = false;
}
bool LookingAtShelter() { return lookAtShelter; }

void LookAtDoodle(bool on)
{
    lookAtDoodle = on;
    if (on)
        lookAtShelter = false;
}
bool LookingAtDoodle() { return lookAtDoodle; }

float CameraCentre() { return cameraX + kScreenW / 2; }

float Ground() { return art.ground; }
float GirlStart() { return art.girlStart; }
float GirlHeight() { return art.girlHeight; }
float NpcHeight() { return art.npcHeight; }
float BuyerScale() { return art.buyerScale; }

float CanvasScale() { return art.clothSize.x > 0 ? art.clothSize.x / kCanvasWidth : 0.3f; }
float CanvasHeight() { return art.clothSize.y / CanvasScale(); }

sa::Vec2 CanvasToScreen(sa::Vec2 canvas)
{
    const sa::Vec2 cloth = ClothScreenPos();
    const float k = CanvasScale();
    return {cloth.x + (canvas.x - kCanvasLeft) * k, cloth.y + (canvas.y - kCanvasTop) * k};
}

sa::Vec2 ScreenToCanvas(sa::Vec2 screen)
{
    const sa::Vec2 cloth = ClothScreenPos();
    const float k = CanvasScale();
    return {kCanvasLeft + (screen.x - cloth.x) / k, kCanvasTop + (screen.y - cloth.y) / k};
}

bool OnCloth(sa::Vec2 screen)
{
    const sa::Vec2 cloth = ClothScreenPos();
    return screen.x >= cloth.x && screen.x <= cloth.x + art.clothSize.x && screen.y >= cloth.y &&
           screen.y <= cloth.y + art.clothSize.y;
}

void DrawStall(sa::Backend& backend, float worldX)
{
    const Object& o = art.stall;
    if (o.texture)
        backend.DrawSprite(o.texture, {}, ToScreen({worldX - o.size.x / 2, o.bottom - o.size.y}), o.size);
}

sa::Vec2 ClothSize() { return art.clothSize.x > 0 ? art.clothSize : sa::Vec2{366.0f, 253.0f}; }

sa::Vec2 ClothScreenPosAt(float worldX)
{
    if (art.clothSize.x <= 0)
        return ToScreen({worldX - ClothSize().x / 2, 183.0f});
    return ToScreen({art.clothPos.x + worldX - art.stall.x, art.clothPos.y});
}

void DrawOnCloth(sa::Backend& backend, const std::vector<std::vector<sa::Vec2>>& strokes, sa::Color color)
{
    const float thickness = std::max(2.5f, kBrush * CanvasScale());
    for (const auto& s : strokes)
    {
        if (s.size() == 1)
            backend.DrawLine(CanvasToScreen(s[0]), CanvasToScreen(s[0]), thickness, color);
        for (size_t i = 1; i < s.size(); ++i)
            backend.DrawLine(CanvasToScreen(s[i - 1]), CanvasToScreen(s[i]), thickness, color);
    }
}

bool DrawRounded(sa::Backend& backend, sa::Vec2 pos, sa::Vec2 size, sa::Color fill, sa::Color edge)
{
    if (!art.fill || !art.edge)
        return false;
    const float c = std::min({kCorner, size.x / 2, size.y / 2});
    // Nine pieces: corners keep their shape, edges stretch one way, the middle both.
    const float xs[4] = {pos.x, pos.x + c, pos.x + size.x - c, pos.x + size.x};
    const float ys[4] = {pos.y, pos.y + c, pos.y + size.y - c, pos.y + size.y};
    const float us[4] = {0.0f, kSlice, 1.0f - kSlice, 1.0f};
    for (sa::TextureId texture : {art.fill, art.edge})
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col)
            {
                const sa::UVRect uv{{us[col], us[row]}, {us[col + 1] - us[col], us[row + 1] - us[row]}};
                backend.DrawSpriteFilled(texture, uv, {xs[col], ys[row]},
                                         {xs[col + 1] - xs[col], ys[row + 1] - ys[row]},
                                         texture == art.fill ? fill : edge);
            }
    return true;
}

namespace
{
// The texture and its nine-slice cut (left, top, right, bottom in texture pixels) for a skin.
std::pair<sa::TextureId, SkinInsets> SkinCut(UiSkin skin)
{
    switch (skin)
    {
    case UiSkin::Panel:       return {art.uiPanel, {62, 38, 68, 40}};
    case UiSkin::Title:       return {art.uiTitle, {150, 65, 150, 48}};
    case UiSkin::Button:      return {art.uiButton, {30, 20, 76, 20}};
    case UiSkin::ButtonGreen: return {art.uiButtonGreen, {72, 22, 32, 22}};
    case UiSkin::ButtonCream: return {art.uiButtonCream, {34, 24, 78, 24}};
    case UiSkin::Speech:      return {art.uiSpeech, {54, 26, 105, 52}};
    }
    return {0, {}};
}
} // namespace

SkinInsets UiSkinInsets(sa::Backend& backend, UiSkin skin, sa::Vec2 size)
{
    const auto [texture, cut] = SkinCut(skin);
    const sa::Vec2 tex = backend.TextureSize(texture);
    if (!texture || tex.x <= 0 || tex.y <= 0)
        return {12, 12, 12, 12};
    const float scale = std::min(1.0f, size.y / tex.y);       // as DrawUiSkin scales the corners
    SkinInsets in{cut.left * scale, cut.top * scale, cut.right * scale, cut.bottom * scale};
    if (in.left + in.right > size.x)
    {
        const float k = size.x / (in.left + in.right);
        in.left *= k;
        in.right *= k;
    }
    if (in.top + in.bottom > size.y)
    {
        const float k = size.y / (in.top + in.bottom);
        in.top *= k;
        in.bottom *= k;
    }
    return in;
}

sa::Vec2 UiSkinFit(sa::Backend& backend, UiSkin skin, sa::Vec2 content)
{
    const auto [texture, cut] = SkinCut(skin);
    const sa::Vec2 tex = backend.TextureSize(texture);
    if (!texture || tex.x <= 0 || tex.y <= 0)
        return {content.x + 24, content.y + 24};
    // Corners shrink with the box below the texture's height: solve h = content + (top + bottom) * h / tex.
    const float vertical = cut.top + cut.bottom;
    float h = content.y + vertical;
    if (h < tex.y)
        h = vertical < tex.y ? content.y / (1.0f - vertical / tex.y) : tex.y;
    const float scale = std::min(1.0f, h / tex.y);
    return {content.x + (cut.left + cut.right) * scale, h};
}

bool DrawUiSkin(sa::Backend& backend, UiSkin skin, sa::Vec2 pos, sa::Vec2 size, float alpha, sa::Color tint)
{
    struct Insets
    {
        float left, top, right, bottom;
    };
    sa::TextureId texture = 0;
    Insets cut{};
    switch (skin)
    {
    case UiSkin::Panel:       texture = art.uiPanel;       cut = {62, 38, 68, 40}; break;
    case UiSkin::Title:       texture = art.uiTitle;       cut = {150, 65, 150, 48}; break;
    case UiSkin::Button:      texture = art.uiButton;      cut = {30, 20, 76, 20}; break;
    case UiSkin::ButtonGreen: texture = art.uiButtonGreen; cut = {72, 22, 32, 22}; break;
    case UiSkin::ButtonCream: texture = art.uiButtonCream; cut = {34, 24, 78, 24}; break;
    case UiSkin::Speech:      texture = art.uiSpeech;      cut = {54, 26, 105, 52}; break;
    }
    const sa::Vec2 tex = backend.TextureSize(texture);
    if (!texture || tex.x <= 0 || tex.y <= 0)
        return false;

    // Keep ornament size tied to the destination height. Very tall panels retain the source size rather
    // than growing giant flowers; short fields shrink cleanly without overlapping their text.
    const float scale = std::min(1.0f, size.y / tex.y);
    float dl = cut.left * scale, dr = cut.right * scale;
    float dt = cut.top * scale, db = cut.bottom * scale;
    if (dl + dr > size.x)
    {
        const float k = size.x / (dl + dr);
        dl *= k;
        dr *= k;
    }
    if (dt + db > size.y)
    {
        const float k = size.y / (dt + db);
        dt *= k;
        db *= k;
    }
    const float xs[4] = {pos.x, pos.x + dl, pos.x + size.x - dr, pos.x + size.x};
    const float ys[4] = {pos.y, pos.y + dt, pos.y + size.y - db, pos.y + size.y};
    const float us[4] = {0.0f, cut.left / tex.x, 1.0f - cut.right / tex.x, 1.0f};
    const float vs[4] = {0.0f, cut.top / tex.y, 1.0f - cut.bottom / tex.y, 1.0f};
    tint.a *= alpha;
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
        {
            const sa::UVRect uv{{us[col], vs[row]}, {us[col + 1] - us[col], vs[row + 1] - vs[row]}};
            backend.DrawSpriteFilled(texture, uv, {xs[col], ys[row]},
                                     {xs[col + 1] - xs[col], ys[row + 1] - ys[row]}, tint);
        }
    return true;
}

} // namespace stage
