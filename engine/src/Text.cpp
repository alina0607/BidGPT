#include "sa/Text.h"

#include <algorithm>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" const unsigned char* sa_font8x8_glyph(int c);

namespace sa
{
namespace
{

// A proportional font from LoadFont (or the fallback font from LoadFallbackFont), at several pixel sizes.
struct Glyph
{
    UVRect uv;
    Vec2 offset;      // from the pen on the baseline to the glyph image's top-left, in atlas pixels
    Vec2 size;        // of the glyph image, in atlas pixels
    float advance = 0;
};

struct FontSize
{
    float px = 0;
    std::unordered_map<char32_t, Glyph> glyphs;
};

struct Font
{
    TextureId texture = 0;
    std::vector<FontSize> sizes;   // ascending

    bool Has(char32_t c) const { return !sizes.empty() && sizes.front().glyphs.count(c) > 0; }

    // The smallest atlas size at least as large as wanted, so glyphs are only ever drawn smaller.
    const FontSize& SizeFor(float px) const
    {
        for (const FontSize& s : sizes)
            if (s.px >= px)
                return s;
        return sizes.back();
    }
};

Font font;        // printable ASCII
Font fallback;    // everything else it has (Chinese for the doodle board's panel)

// The font is drawn this many pixels high per unit of scale: at scale 2 the letters are 20 px, about the
// height of the 16-px bitmap glyphs with their spacing.
constexpr float kPixelsPerScale = 10.0f;
// Where the baseline sits in the kGlyphSize × scale box, as a share of the font's pixel size.
constexpr float kBaseline = 0.76f;

// The code point starting at text[i], moving i past it; a malformed UTF-8 byte reads as '?'.
char32_t NextCodePoint(std::string_view text, size_t& i)
{
    const unsigned char lead = static_cast<unsigned char>(text[i++]);
    if (lead < 0x80)
        return lead;
    const int extra = lead >= 0xF0 ? 3 : lead >= 0xE0 ? 2 : lead >= 0xC0 ? 1 : -1;
    if (extra < 0)
        return U'?';
    if (i + extra > text.size())
    {
        i = text.size();            // cut off at the end of the text
        return U'?';
    }
    char32_t c = lead & (0x3F >> extra);
    for (int k = 0; k < extra; ++k)
    {
        const unsigned char next = static_cast<unsigned char>(text[i]);
        if ((next & 0xC0) != 0x80)
            return U'?';
        c = (c << 6) | (next & 0x3F);
        ++i;
    }
    return c;
}

// The font to draw `c` with and its glyph at `px` (with the scale from atlas to screen pixels): ASCII from the
// main font, anything else from the fallback font when it has it, otherwise the main font's '?'.
struct Placed
{
    const Font* font;
    const Glyph* glyph;
    float k;
};

Placed Place(char32_t c, float px)
{
    const Font& f = !font.Has(c) && fallback.Has(c) ? fallback : font;
    const FontSize& size = f.SizeFor(px);
    static const Glyph none{};
    if (const auto it = size.glyphs.find(c); it != size.glyphs.end())
        return {&f, &it->second, px / size.px};
    const FontSize& main = font.SizeFor(px);
    const auto it = main.glyphs.find(U'?');
    return {&font, it != main.glyphs.end() ? &it->second : &none, px / main.px};
}

bool ReadFont(Backend& backend, const std::string& atlasPath, const std::string& metricsPath, bool asciiOnly,
              Font& out)
{
    std::ifstream in(metricsPath);
    const TextureId texture = in ? backend.LoadTexture(atlasPath, TextureFilter::Smooth) : 0;
    const Vec2 atlas = backend.TextureSize(texture);
    if (!texture || atlas.x <= 0)
        return false;
    Font loaded{texture, {}};
    std::string word;
    while (in >> word)
    {
        if (word == "size")
        {
            float px = 0, ascent = 0, height = 0;
            in >> px >> ascent >> height;
            loaded.sizes.push_back({px, {}});
            continue;
        }
        const long code = std::stol(word);
        float x, y, w, h, left, top, advance;
        in >> x >> y >> w >> h >> left >> top >> advance;
        if (loaded.sizes.empty() || code < 32 || code > 0x10FFFF || (asciiOnly && code > 126))
            return false;
        loaded.sizes.back().glyphs[static_cast<char32_t>(code)] = {
            {{x / atlas.x, y / atlas.y}, {w / atlas.x, h / atlas.y}}, {left, top}, {w, h}, advance};
    }
    if (loaded.sizes.empty())
        return false;
    out = std::move(loaded);
    return true;
}

} // namespace

bool LoadFont(Backend& backend, const std::string& atlasPath, const std::string& metricsPath)
{
    return ReadFont(backend, atlasPath, metricsPath, true, font);
}

bool LoadFallbackFont(Backend& backend, const std::string& atlasPath, const std::string& metricsPath)
{
    return ReadFont(backend, atlasPath, metricsPath, false, fallback);
}

float TextWidth(std::string_view text, float scale)
{
    float width = 0;
    const float px = kPixelsPerScale * scale;
    for (size_t i = 0; i < text.size();)
    {
        const char32_t c = NextCodePoint(text, i);
        if (font.sizes.empty())
            width += kGlyphSize * scale;
        else
        {
            const Placed p = Place(c, px);
            width += p.glyph->advance * p.k;
        }
    }
    return width;
}

float LineHeight(float scale) { return kGlyphSize * scale * kLineSpacing; }

void DrawText(Backend& backend, Vec2 topLeft, std::string_view text, float scale, Color color)
{
    if (!font.sizes.empty())
    {
        const float px = kPixelsPerScale * scale;
        Vec2 pen{topLeft.x, topLeft.y + kBaseline * px};
        for (size_t i = 0; i < text.size();)
        {
            const char32_t c = NextCodePoint(text, i);
            const Placed p = Place(c, px);
            const Glyph& g = *p.glyph;
            if (c != U' ')
                backend.DrawSpriteFilled(p.font->texture, g.uv, {pen.x + g.offset.x * p.k, pen.y + g.offset.y * p.k},
                                         {g.size.x * p.k, g.size.y * p.k}, color);
            pen.x += g.advance * p.k;
        }
        return;
    }
    float x = topLeft.x;
    for (size_t i = 0; i < text.size();)
    {
        const char32_t c = NextCodePoint(text, i);
        const unsigned char* rows = sa_font8x8_glyph(c < 128 ? static_cast<int>(c) : '?');
        for (int r = 0; r < 8; ++r)
        {
            // Merge runs of lit pixels in a row into one rectangle: far fewer quads per glyph.
            int c0 = 0;
            while (c0 < 8)
            {
                if (!((rows[r] >> c0) & 1))
                {
                    ++c0;
                    continue;
                }
                int c1 = c0;
                while (c1 < 8 && ((rows[r] >> c1) & 1))
                    ++c1;
                backend.DrawRect({x + c0 * scale, topLeft.y + r * scale},
                                 {(c1 - c0) * scale, scale}, color);
                c0 = c1;
            }
        }
        x += kGlyphSize * scale;
    }
}

std::vector<std::string> WrapText(std::string_view text, float maxWidth, float scale)
{
    // Words are runs of ASCII between spaces; every other character (Chinese has no spaces) is a word of its
    // own that joins its neighbours without a space, so a line may break between any two of them.
    std::vector<std::string> lines;
    std::string current;
    bool spaceBefore = false;      // a space came between `current` and the next word
    size_t i = 0;
    while (i < text.size())
    {
        if (text[i] == '\n')
        {
            lines.push_back(current);
            current.clear();
            spaceBefore = false;
            ++i;
            continue;
        }
        if (text[i] == ' ')
        {
            spaceBefore = !current.empty();
            ++i;
            continue;
        }
        size_t end = i;
        const bool ascii = static_cast<unsigned char>(text[i]) < 0x80;
        if (ascii)
            while (end < text.size() && text[end] != ' ' && text[end] != '\n' &&
                   static_cast<unsigned char>(text[end]) < 0x80)
                ++end;
        else
            NextCodePoint(text, end);
        std::string_view word = text.substr(i, end - i);
        i = end;
        // A word longer than a whole line is split hard, keeping at least one character per line.
        while (ascii && TextWidth(word, scale) > maxWidth && word.size() > 1)
        {
            if (!current.empty())
            {
                lines.push_back(current);
                current.clear();
            }
            size_t fit = 1;
            while (fit < word.size() && TextWidth(word.substr(0, fit + 1), scale) <= maxWidth)
                ++fit;
            lines.emplace_back(word.substr(0, fit));
            word.remove_prefix(fit);
            spaceBefore = false;
        }
        const std::string joined = current + (spaceBefore ? " " : "") + std::string(word);
        // Chinese closing punctuation never starts a line: it hangs past the end of the line before.
        size_t at = 0;
        const char32_t c = ascii ? 0 : NextCodePoint(word, at);
        const bool closing = c == U'\u3001' || c == U'\u3002' || c == U'\uFF0C' || c == U'\uFF1A' ||
                             c == U'\uFF1B' || c == U'\uFF09' || c == U'\uFF01' || c == U'\uFF1F';
        if (!current.empty() && !closing && TextWidth(joined, scale) > maxWidth)
        {
            lines.push_back(current);
            current = std::string(word);
        }
        else
            current = joined;
        spaceBefore = false;
    }
    if (!current.empty() || lines.empty())
        lines.push_back(current);
    return lines;
}

float DrawWrapped(Backend& backend, Vec2 topLeft, std::string_view text, float maxWidth, float scale,
                  Color color)
{
    float y = topLeft.y;
    for (const std::string& line : WrapText(text, maxWidth, scale))
    {
        DrawText(backend, {topLeft.x, y}, line, scale, color);
        y += LineHeight(scale);
    }
    return y - topLeft.y;
}

} // namespace sa
