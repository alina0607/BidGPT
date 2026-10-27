#pragma once

#include "sa/Backend.h"

#include <vector>

// The world every screen is drawn on (docs/art.md): an endless watercolor strip, the objects standing in
// it (the stall with its cloth, the cat shelter; see assets/stage/layout.json). The girl the player walks
// around is in Girl.h. World x is 0 at the stall's centre; world y is screen y. Loaded on first use;
// without the art the game still runs on the plain background.

namespace stage
{

enum class UiSkin
{
    Panel,
    Title,
    Button,
    ButtonGreen,
    ButtonCream,
    Speech,
};

// The drawing canvas the game keeps strokes in (the size the classifier and the AI drawings were made for),
// shown on the stall's cloth.
constexpr float kCanvasLeft = 20.0f;
constexpr float kCanvasTop = 70.0f;
constexpr float kCanvasWidth = 1240.0f;

// Loads everything once; safe to call every frame.
void Load(sa::Backend& backend);
// Moves the camera smoothly once world x `x` (the girl) leaves the middle of the screen.
void Follow(float x, float dt);
sa::Vec2 ToScreen(sa::Vec2 world);
// The background and the objects (the girl and the buyers are drawn by their owners, after this).
void DrawWorld(sa::Backend& backend);
// The cat shelter's easels (left to right), blank until something is painted on them: PaintEasel puts an RGBA
// picture (rows top to bottom) on easel i, replacing what was there; ClearEasels blanks them all.
int EaselCount();
void ClearEasels();
void PaintEasel(sa::Backend& backend, int i, int width, int height, const unsigned char* rgba);
// Easel i's painting: its texture (0 while the easel is blank) and where it is on screen (top-left, size).
sa::TextureId EaselPainting(int i);
void EaselScreenRect(int i, sa::Vec2& topLeft, sa::Vec2& size);
// The painted easel under a screen point, or -1.
int EaselAt(sa::Vec2 screen);
// While on, the camera shows the whole shelter instead of following the girl.
void LookAtShelter(bool on);
bool LookingAtShelter();
// The colour cat doodle station, left of the stall (DoodleBoard.h): a copy of the stall whose cloth is the doodle
// board, with its tools and reading panel, centred on world x kDoodleX, and another copy of the stall, its cloth
// left for Brush GPT to paint on later, centred on kFrameX.
constexpr float kDoodleX = -1800.0f;
constexpr float kFrameX = -820.0f;
// While on, the camera shows the doodle station instead of following the girl (off again: LookAtShelter turns
// it off, and so does the caller once she walks away).
void LookAtDoodle(bool on);
bool LookingAtDoodle();
// The world x at the middle of the screen.
float CameraCentre();
// The screen y the girl stands on, and the buyers' sprite scale.
float Ground();
float BuyerScale();
float GirlStart();          // where she stands when the game begins: beside the cloth, not in front of it
float GirlHeight();         // her height on screen, standing on the ground line
float NpcHeight();          // a painted buyer's height when standing at the stall

// The canvas on the cloth: its height in canvas units, and points both ways.
float CanvasHeight();
float CanvasScale();                    // screen pixels per canvas unit
sa::Vec2 CanvasToScreen(sa::Vec2 canvas);
sa::Vec2 ScreenToCanvas(sa::Vec2 screen);
bool OnCloth(sa::Vec2 screen);
// The stall drawn again with its stall at world x (the doodle station's copies), and that copy's cloth on screen:
// its top-left corner and its size (the same for every copy; without the art, a plain cloth-sized rectangle).
void DrawStall(sa::Backend& backend, float worldX);
sa::Vec2 ClothScreenPosAt(float worldX);
sa::Vec2 ClothSize();
// Strokes in canvas units, drawn on the cloth.
void DrawOnCloth(sa::Backend& backend, const std::vector<std::vector<sa::Vec2>>& strokes, sa::Color color);

// A rounded rectangle: fill, then a thin edge. False without the art (the caller draws a plain one).
bool DrawRounded(sa::Backend& backend, sa::Vec2 pos, sa::Vec2 size, sa::Color fill, sa::Color edge);

// Watercolour-and-flower UI pieces from assets/ui. Each is drawn as a nine-slice so the flowers and
// rounded corners keep their shape while the paper centre stretches to fit dynamic text.
bool DrawUiSkin(sa::Backend& backend, UiSkin skin, sa::Vec2 pos, sa::Vec2 size, float alpha = 1.0f,
                sa::Color tint = {1, 1, 1, 1});

// Room the skin's frame and flowers take on each side when drawn at `size` (the same corner sizes
// DrawUiSkin uses), so text can sit on the paper only. A plain 12px border without the art.
struct SkinInsets
{
    float left, top, right, bottom;
};
SkinInsets UiSkinInsets(sa::Backend& backend, UiSkin skin, sa::Vec2 size);
// The size a skinned box needs so that `content` fits inside its frame and flowers.
sa::Vec2 UiSkinFit(sa::Backend& backend, UiSkin skin, sa::Vec2 content);

} // namespace stage
