// Sprite sheet cells, animation timing, and texture loading on the headless backend.

#include "sa/Backend.h"
#include "sa/Sprite.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace
{

int failures = 0;

void Check(bool ok, const char* what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    failures += ok ? 0 : 1;
}

bool Near(float a, float b) { return std::fabs(a - b) < 1e-6f; }

} // namespace

int main(int argc, char** argv)
{
    // The CS529 monkey's walk: a 3 × 3 sheet with 8 frames in use.
    const sa::SpriteSheet walk{0, 3, 3, 8};
    const sa::UVRect c4 = walk.Cell(4);
    Check(Near(c4.offset.x, 1.0f / 3) && Near(c4.offset.y, 1.0f / 3) && Near(c4.size.x, 1.0f / 3),
          "frame 4 of a 3x3 sheet is the middle cell");
    const sa::UVRect c7 = walk.Cell(7);
    Check(Near(c7.offset.x, 1.0f / 3) && Near(c7.offset.y, 2.0f / 3), "frame 7 is row 3, column 2");
    Check(Near(walk.Cell(99).offset.x, c7.offset.x) && Near(walk.Cell(-5).offset.x, 0.0f),
          "frames out of range are clamped to the ones in use");

    sa::Animation loop;
    loop.Play(8, 0.05f, true);
    loop.Update(0.12f);
    Check(loop.Frame() == 2, "after 0.12 s at 0.05 s a frame: frame 2");
    loop.Update(0.33f);
    Check(loop.Frame() == 1 && !loop.Done(), "a looping animation wraps around (0.45 s: frame 9 mod 8 = 1)");

    sa::Animation once;
    once.Play(6, 0.08f, false);
    once.Update(0.47f);
    Check(once.Frame() == 5 && !once.Done(), "a one-shot animation reaches its last frame");
    once.Update(0.02f);
    Check(once.Frame() == 5 && once.Done(), "and is done after showing it for one frame time");

    // Textures load (and fail) the same way headless, so CI notices broken art.
    const std::string dir = argc > 1 ? argv[1] : "engine/tests/fixtures";
    auto backend = sa::CreateBackend("null");
    const sa::TextureId sheet = backend->LoadTexture(dir + "/sheet_3x1.png");
    const sa::Vec2 size = backend->TextureSize(sheet);
    Check(sheet != 0 && Near(size.x, 12) && Near(size.y, 4), "a 12x4 PNG loads with its size");
    Check(backend->LoadTexture(dir + "/missing.png") == 0, "a missing file gives texture 0");

    std::printf(failures ? "%d failure(s)\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
