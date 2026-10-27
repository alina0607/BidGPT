// Unit tests for text layout (no window needed). Exits non-zero on the first failure.
#include "sa/Text.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace
{

int failures = 0;

void Check(bool ok, const char* what)
{
    if (!ok)
    {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

// Counts rectangles instead of drawing them.
struct CountingBackend final : sa::Backend
{
    int rects = 0;
    std::string_view Name() const override { return "counting"; }
    bool Init(const sa::WindowDesc&) override { return true; }
    void Shutdown() override {}
    bool BeginFrame(sa::Color) override { return true; }
    void EndFrame() override {}
    sa::Vec2 MousePosition() const override { return {}; }
    bool MouseDown(sa::MouseButton) const override { return false; }
    bool KeyPressed(sa::Key) const override { return false; }
    void DrawLine(sa::Vec2, sa::Vec2, float, sa::Color) override {}
    void DrawRect(sa::Vec2, sa::Vec2, sa::Color) override { ++rects; }
};

// Pretends every texture is a 100×100 atlas and counts glyphs drawn in a color.
struct FontBackend final : sa::Backend
{
    int glyphs = 0;
    sa::Color last{};
    std::string_view Name() const override { return "font"; }
    bool Init(const sa::WindowDesc&) override { return true; }
    void Shutdown() override {}
    bool BeginFrame(sa::Color) override { return true; }
    void EndFrame() override {}
    sa::Vec2 MousePosition() const override { return {}; }
    bool MouseDown(sa::MouseButton) const override { return false; }
    bool KeyPressed(sa::Key) const override { return false; }
    void DrawLine(sa::Vec2, sa::Vec2, float, sa::Color) override {}
    void DrawRect(sa::Vec2, sa::Vec2, sa::Color) override {}
    sa::TextureId LoadTexture(const std::string&, sa::TextureFilter) override { return 1; }
    sa::Vec2 TextureSize(sa::TextureId) const override { return {100, 100}; }
    void DrawSpriteFilled(sa::TextureId, sa::UVRect, sa::Vec2, sa::Vec2, sa::Color color) override
    {
        ++glyphs;
        last = color;
    }
};

// A one-size font, 10 px: ' ' and 'i' 3 px wide, 'W' 10 px.
std::string WriteMetrics()
{
    const auto path = std::filesystem::temp_directory_path() / "sa_text_test_font.txt";
    std::ofstream out(path);
    out << "size 10 10 13\n32 0 0 2 2 0 0 3\n105 4 0 4 12 0 -10 3\n87 10 0 12 12 0 -10 10\n";
    return path.string();
}

// A fallback font with two Chinese characters, 10 px wide each: 貓 (U+8C93) and 是 (U+662F), and 、 (U+3001), 5 px.
std::string WriteFallbackMetrics()
{
    const auto path = std::filesystem::temp_directory_path() / "sa_text_test_cjk.txt";
    std::ofstream out(path);
    out << "size 10 10 13\n35987 20 0 10 10 0 -9 10\n26159 30 0 10 10 0 -9 10\n12289 40 0 5 5 0 -3 5\n";
    return path.string();
}

} // namespace

int main()
{
    // 8 px glyphs at scale 2 → 16 px per character.
    Check(sa::TextWidth("hello", 2.0f) == 80.0f, "TextWidth is chars × 8 × scale");

    // 10 characters per line at scale 1 with maxWidth 80.
    auto lines = sa::WrapText("the quick brown fox jumps", 80.0f, 1.0f);
    Check(lines == std::vector<std::string>{"the quick", "brown fox", "jumps"}, "wraps at spaces");

    lines = sa::WrapText("abcdefghijklmnop", 80.0f, 1.0f);
    Check(lines == std::vector<std::string>{"abcdefghij", "klmnop"}, "hard-splits words longer than a line");

    lines = sa::WrapText("one\ntwo", 800.0f, 1.0f);
    Check(lines == std::vector<std::string>{"one", "two"}, "honors newlines");

    Check(sa::WrapText("", 80.0f, 1.0f).size() == 1, "empty text is one empty line");

    // UTF-8: one glyph per character, not per byte (貓 is three bytes).
    Check(sa::TextWidth("\xE8\xB2\x93" "a", 1.0f) == 16.0f, "a multi-byte character is one bitmap glyph wide");
    Check(sa::TextWidth("\xE8\xB2", 1.0f) == 8.0f, "a cut-off character reads as one '?'");

    CountingBackend b;
    sa::DrawText(b, {0, 0}, " ", 1.0f, {});
    Check(b.rects == 0, "space draws nothing");
    sa::DrawText(b, {0, 0}, "_", 1.0f, {});
    Check(b.rects == 1, "underscore is one merged run");
    b.rects = 0;
    sa::DrawText(b, {0, 0}, "A", 1.0f, {});
    Check(b.rects > 0 && b.rects <= 14, "'A' merges each row into at most two runs");

    // A proportional font replaces the bitmap one for everything above.
    FontBackend f;
    Check(!sa::LoadFont(f, "atlas.png", "/no/such/metrics.txt"), "LoadFont fails without its metrics");
    Check(sa::TextWidth("WWW", 1.0f) == 24.0f, "a failed load keeps the bitmap font");
    Check(sa::LoadFont(f, "atlas.png", WriteMetrics()), "LoadFont reads the metrics");
    Check(sa::TextWidth("iii", 1.0f) == 9.0f && sa::TextWidth("WWW", 1.0f) == 30.0f,
          "widths add the glyphs' advances");
    Check(sa::TextWidth("W", 2.0f) == 20.0f, "larger text scales the largest atlas size up");
    Check(sa::WrapText("ii ii", 12.0f, 1.0f) == std::vector<std::string>{"ii", "ii"}, "wraps by measured width");
    Check(sa::WrapText("WWW", 20.0f, 1.0f) == std::vector<std::string>{"WW", "W"}, "hard-splits by measured width");
    const sa::Color pink{1.0f, 0.5f, 0.6f, 1.0f};
    sa::DrawText(f, {0, 0}, "i i", 1.0f, pink);
    Check(f.glyphs == 2 && f.last.g == 0.5f, "one filled glyph per non-space character, in the text's color");

    // Chinese comes from the fallback font; a character neither font has is the main font's '?'.
    Check(!sa::LoadFallbackFont(f, "cjk.png", "/no/such/metrics.txt"), "LoadFallbackFont fails without its metrics");
    Check(sa::TextWidth("\xE8\xB2\x93", 1.0f) == 0.0f, "without the fallback it is the main font's '?' (none in this font: no width)");
    Check(sa::LoadFallbackFont(f, "cjk.png", WriteFallbackMetrics()), "LoadFallbackFont reads the metrics");
    const std::string cat = "\xE8\xB2\x93", yes = "\xE6\x98\xAF";       // 貓, 是
    Check(sa::TextWidth(cat + yes + "i", 1.0f) == 23.0f, "fallback glyphs and ASCII glyphs add up");
    Check(sa::TextWidth("i", 1.0f) == 3.0f, "ASCII still comes from the main font");
    Check(sa::WrapText(cat + yes + cat + yes, 20.0f, 1.0f) == std::vector<std::string>{cat + yes, cat + yes},
          "Chinese breaks between characters without spaces");
    Check(sa::WrapText("ii " + cat + yes, 16.0f, 1.0f) == std::vector<std::string>{"ii", cat, yes},
          "a space before Chinese is kept as a break");
    const std::string comma = "\xE3\x80\x81";       // 、
    Check(sa::WrapText(cat + yes + comma + cat, 20.0f, 1.0f) == std::vector<std::string>{cat + yes + comma, cat},
          "Chinese closing punctuation stays at the end of the line");
    f.glyphs = 0;
    sa::DrawText(f, {0, 0}, cat + " " + yes, 1.0f, pink);
    Check(f.glyphs == 2, "one glyph per Chinese character");

    std::printf(failures ? "%d failure(s)\n" : "all text tests passed\n", failures);
    return failures ? 1 : 0;
}
