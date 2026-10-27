#include "DoodleBoard.h"

#include "Stage.h"

#include "sa/Text.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <optional>
#include <utility>

namespace
{

constexpr float kScreenW = 1280.0f;
constexpr float kDebounce = 0.35f;      // s after a stroke before it is read: a quick next stroke joins the request
constexpr float kMinSpacing = 2.0f;     // paper pixels between the points of a stroke

// Where things are, in "station" coordinates: screen pixels while the camera is centred on the station.
constexpr sa::Vec2 kToolsPos{12, 136}, kToolsSize{180, 482};
constexpr float kStallX = 486.0f;       // the stall copy's centre: the stall spans about 215..758
constexpr sa::Vec2 kPanelPos{772, 136}, kPanelSize{500, 482};
constexpr float kToolX = 22.0f, kToolStep = 82.0f;
constexpr float kSwatchY = 168.0f, kSwatchW = 74.0f, kSwatchH = 30.0f, kSwatchStep = 36.0f;
constexpr float kPenY = 376.0f, kPenW = 74.0f, kPenH = 52.0f;
constexpr float kSizeY = 518.0f, kSizeH = 36.0f;
constexpr sa::Vec2 kClearPos{22, 566}, kClearSize{156, 42};
constexpr sa::Vec2 kSavePos{834, 556}, kSaveSize{300, 46};     // in the reading panel

struct Pigment
{
    const char* name;
    std::uint8_t r, g, b;
};

// The 10 paints: only colour families Brush GPT has learnt to paint from real cats (owner, 2026-10-17). Measured on
// 3,000 of run J's training paintings: red and purple are the nearest paint for 0.40% and 0.17% of strokes, so they
// were taken off the board (plan.md D25); pink is noses and ears, yellow, green and blue are eyes. The reader's fur
// colours (white ... brown, kCoats) are shown in the panel with the paint that looks like them.
constexpr std::array<Pigment, 10> kPalette{{
    {"黑", 30, 28, 28},     {"白", 250, 250, 247},   {"灰", 138, 138, 138},  {"橘", 232, 128, 38},
    {"奶油", 241, 214, 160}, {"棕", 117, 74, 42},     {"黃", 242, 204, 48},   {"綠", 72, 160, 80},
    {"藍", 62, 112, 204},    {"粉紅", 242, 152, 182},
}};
// The paints' names in strokes.json.
constexpr std::array<const char*, 10> kPaintCodes{"black",  "white", "grey", "orange", "cream",
                                                  "brown",  "yellow", "green", "blue",  "pink"};
// The reader's paper (its training doodles' beige; the stall's cloth averages (247, 230, 213)): white paint and bare
// paper differ.
constexpr std::uint8_t kPaper[3] = {0xED, 0xE6, 0xD6};

struct Name
{
    const char* code;
    const char* zh;
    std::uint8_t r, g, b;
};

constexpr std::array<Name, 7> kCoats{{{"white", "白", 250, 250, 247}, {"black", "黑", 30, 28, 28},
                                      {"grey", "灰", 138, 138, 138},   {"orange", "橘", 232, 128, 38},
                                      {"cream", "奶油", 241, 214, 160}, {"brown", "棕", 117, 74, 42},
                                      {"mixed", "混色", 0, 0, 0}}};
constexpr std::array<Name, 5> kPatterns{{{"solid", "純色", 0, 0, 0}, {"tabby", "虎斑", 0, 0, 0},
                                         {"spots", "斑點", 0, 0, 0}, {"bicolour", "雙色", 0, 0, 0},
                                         {"tortie_calico", "玳瑁／三花", 0, 0, 0}}};
constexpr std::array<Name, 5> kEyes{{{"yellow", "黃", 240, 200, 40}, {"green", "綠", 80, 170, 80},
                                     {"blue", "藍", 70, 130, 210},   {"orange", "橘", 232, 128, 38},
                                     {"copper", "古銅", 184, 108, 50}}};
// Left and right as seen on the screen, the reader's convention.
constexpr std::array<std::pair<const char*, const char*>, 7> kRegions{{{"crown", "頭頂"},
                                                                       {"ear_left", "左耳"},
                                                                       {"ear_right", "右耳"},
                                                                       {"bridge", "鼻樑"},
                                                                       {"muzzle_chin", "口鼻和下巴"},
                                                                       {"cheek_left", "左臉頰"},
                                                                       {"cheek_right", "右臉頰"}}};
constexpr const char* kUnspecified = "未指定";

template <size_t N>
const Name* Find(const std::array<Name, N>& names, const std::string& code)
{
    for (const Name& n : names)
        if (code == n.code)
            return &n;
    return nullptr;
}

// A string field of the reading, or nothing for null, a missing key or anything that is not a name.
std::optional<std::string> Field(const nlohmann::json& object, const char* key)
{
    if (!object.is_object() || !object.contains(key) || !object[key].is_string())
        return std::nullopt;
    return object[key].get<std::string>();
}

// Half the brush's width, in paint pixels (about 2.1 to a screen pixel on the cloth).
float HalfWidth(DoodleBoard::Tool tool, bool thick)
{
    switch (tool)
    {
    case DoodleBoard::Tool::Pencil: return thick ? 3.2f : 1.7f;
    case DoodleBoard::Tool::Flat:   return thick ? 18.0f : 8.0f;
    case DoodleBoard::Tool::Taper:  return thick ? 11.0f : 5.0f;
    case DoodleBoard::Tool::Eraser: return thick ? 26.0f : 10.0f;
    }
    return 2.0f;
}

const char* ToolName(DoodleBoard::Tool tool)
{
    switch (tool)
    {
    case DoodleBoard::Tool::Pencil: return "鉛筆";
    case DoodleBoard::Tool::Flat:   return "平筆";
    case DoodleBoard::Tool::Taper:  return "尖筆";
    case DoodleBoard::Tool::Eraser: return "橡皮擦";
    }
    return "";
}

const char* ToolCode(DoodleBoard::Tool tool)
{
    switch (tool)
    {
    case DoodleBoard::Tool::Pencil: return "pencil";
    case DoodleBoard::Tool::Flat:   return "flat";
    case DoodleBoard::Tool::Taper:  return "taper";
    case DoodleBoard::Tool::Eraser: return "eraser";
    }
    return "";
}

const char* PaintCode(int colour) { return kPaintCodes[static_cast<size_t>(colour)]; }

// Fixed per-pixel grain for the pencil (the same every frame, so a line does not shimmer while it grows).
float Grain(int x, int y)
{
    std::uint32_t h = static_cast<std::uint32_t>(x) * 374761393u + static_cast<std::uint32_t>(y) * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return static_cast<float>((h ^ (h >> 16)) & 0xFFFFu) / 65535.0f;
}

std::string Base64(const std::vector<std::uint8_t>& bytes)
{
    static const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((bytes.size() + 2) / 3 * 4);
    for (size_t i = 0; i < bytes.size(); i += 3)
    {
        const unsigned n = static_cast<unsigned>(bytes[i]) << 16 |
                           (i + 1 < bytes.size() ? static_cast<unsigned>(bytes[i + 1]) << 8 : 0u) |
                           (i + 2 < bytes.size() ? static_cast<unsigned>(bytes[i + 2]) : 0u);
        out += alphabet[(n >> 18) & 63];
        out += alphabet[(n >> 12) & 63];
        out += i + 1 < bytes.size() ? alphabet[(n >> 6) & 63] : '=';
        out += i + 2 < bytes.size() ? alphabet[n & 63] : '=';
    }
    return out;
}

float Distance(sa::Vec2 p, sa::Vec2 a, sa::Vec2 b)
{
    const sa::Vec2 ab = b - a, ap = p - a;
    const float len2 = ab.x * ab.x + ab.y * ab.y;
    const float t = len2 > 0 ? std::clamp((ap.x * ab.x + ap.y * ab.y) / len2, 0.0f, 1.0f) : 0.0f;
    const sa::Vec2 d{ap.x - ab.x * t, ap.y - ab.y * t};
    return std::sqrt(d.x * d.x + d.y * d.y);
}

sa::Color Rgb(std::uint8_t r, std::uint8_t g, std::uint8_t b)
{
    return {r / 255.0f, g / 255.0f, b / 255.0f, 1.0f};
}

bool Inside(sa::Vec2 p, sa::Vec2 pos, sa::Vec2 size)
{
    return p.x >= pos.x && p.x <= pos.x + size.x && p.y >= pos.y && p.y <= pos.y + size.y;
}

void Outline(sa::Backend& b, sa::Vec2 pos, sa::Vec2 size, float w, sa::Color c)
{
    b.DrawRect({pos.x - w, pos.y - w}, {size.x + 2 * w, w}, c);
    b.DrawRect({pos.x - w, pos.y + size.y}, {size.x + 2 * w, w}, c);
    b.DrawRect({pos.x - w, pos.y}, {w, size.y}, c);
    b.DrawRect({pos.x + size.x, pos.y}, {w, size.y}, c);
}

void Swatch(sa::Backend& b, sa::Vec2 pos, sa::Vec2 size, sa::Color c)
{
    b.DrawRect({pos.x - 1, pos.y - 1}, {size.x + 2, size.y + 2}, ui::kBorder);
    b.DrawRect(pos, size, c);
}

sa::Vec2 SwatchPos(int i) { return {kToolX + (i % 2) * kToolStep, kSwatchY + (i / 2) * kSwatchStep}; }
sa::Vec2 PenPos(int i) { return {kToolX + (i % 2) * kToolStep, kPenY + (i / 2) * 58.0f}; }
sa::Vec2 SizePos(int i) { return {kToolX + i * kToolStep, kSizeY}; }

// Station coordinates on screen: where the station's left edge is now (the camera may be anywhere).
sa::Vec2 Origin() { return {stage::ToScreen({stage::kDoodleX - kScreenW / 2, 0}).x, 0}; }
float StallWorldX() { return stage::kDoodleX - kScreenW / 2 + kStallX; }

// The cloth of the doodle stall on screen, and the paint pixel under a screen point.
sa::Vec2 ClothPos() { return stage::ClothScreenPosAt(StallWorldX()); }
sa::Vec2 ToPaint(sa::Vec2 screen)
{
    const sa::Vec2 c = ClothPos(), size = stage::ClothSize();
    return {(screen.x - c.x) * DoodleBoard::kResW / size.x, (screen.y - c.y) * DoodleBoard::kResH / size.y};
}
bool OnCloth(sa::Vec2 screen) { return Inside(screen, ClothPos(), stage::ClothSize()); }

// The reader's square: the paint centred on it (no padding while the cloth is square).
constexpr int kPadTop = (DoodleBoard::kResW - DoodleBoard::kResH) / 2;
static_assert(kPadTop >= 0, "the paint must not be taller than it is wide");

} // namespace

DoodleBoard::DoodleBoard(std::string host, int port)
    : client_(std::move(host), port),
      paper_(static_cast<size_t>(kResW) * kResH * 4),
      cover_(static_cast<size_t>(kResW) * kResH),
      phase_(static_cast<size_t>(kResW) * kResH)
{
    Clear();
}

void DoodleBoard::Clear()
{
    std::fill(paper_.begin(), paper_.end(), std::uint8_t{0});      // nothing painted: the cloth shows
    ink_.clear();
    history_.clear();
    anything_ = false;
    penDown_ = false;
    current_ = {};
    dirty_ = false;
    reading_ = {};
    error_.clear();
    examStatus_.clear();
    ++generation_;              // a reading still on its way is for the old drawing
    changed_ = true;
}

bool DoodleBoard::InView() const { return std::abs(Origin().x) < kScreenW * 0.5f; }

bool DoodleBoard::Contains(sa::Vec2 screen) const
{
    const sa::Vec2 s{screen.x - Origin().x, screen.y};
    return Inside(s, kToolsPos, kToolsSize) || OnCloth(screen) || Inside(s, kPanelPos, kPanelSize);
}

void DoodleBoard::Update(sa::Backend& backend, const ui::Mouse& mouse, float dt)
{
    if (form_.open)
        UpdateForm(backend, mouse);
    else
    {
        const sa::Vec2 station{mouse.Position().x - Origin().x, mouse.Position().y};
        if (mouse.JustPressed() && !penDown_)
            TakeClick(station);
        TrackPen(mouse, backend.MouseDown(sa::MouseButton::Left));
    }
    Show(backend);
    sinceStroke_ += dt;
    Poll();
    if (examQueued_ && !client_.Busy())
        SendExam();
    else if (dirty_ && !penDown_ && sinceStroke_ >= kDebounce && !client_.Busy())
        SendReading();
}

bool DoodleBoard::Current() const
{
    return reading_.have && !dirty_ && !penDown_ && awaiting_ != Awaiting::Reading;
}

bool DoodleBoard::HasPencil() const
{
    return std::any_of(ink_.begin(), ink_.end(), [](const Stroke& s) { return s.tool == Tool::Pencil; });
}

void DoodleBoard::TakeClick(sa::Vec2 s)
{
    for (int i = 0; i < static_cast<int>(kPalette.size()); ++i)
        if (Inside(s, SwatchPos(i), {kSwatchW, kSwatchH}))
        {
            colour_ = i;
            if (tool_ == Tool::Eraser)
                tool_ = Tool::Pencil;       // picking a paint means painting again
        }
    for (int i = 0; i < 4; ++i)
        if (Inside(s, PenPos(i), {kPenW, kPenH}))
            tool_ = static_cast<Tool>(i);
    for (int i = 0; i < 2; ++i)
        if (Inside(s, SizePos(i), {kPenW, kSizeH}))
            thick_ = i == 1;
    if (anything_ && Inside(s, kClearPos, kClearSize))
        Clear();
    if (Current() && !examQueued_ && awaiting_ != Awaiting::Exam && Inside(s, kSavePos, kSaveSize))
        OpenForm();
}

// A press on the paper starts a stroke; dragging extends it, also off the paper (the paint stops at its edge).
void DoodleBoard::TrackPen(const ui::Mouse& mouse, bool down)
{
    const sa::Vec2 p = ToPaint(mouse.Position());
    if (!penDown_ && down && mouse.JustPressed() && OnCloth(mouse.Position()))
    {
        penDown_ = true;
        current_ = {tool_, colour_, thick_, {p}};
        changed_ = true;
    }
    else if (penDown_ && down)
    {
        const sa::Vec2 d = p - current_.points.back();
        if (std::sqrt(d.x * d.x + d.y * d.y) >= kMinSpacing)
        {
            current_.points.push_back(p);
            changed_ = true;
        }
    }
    else if (penDown_)
        EndStroke();
}

void DoodleBoard::EndStroke()
{
    penDown_ = false;
    Paint(current_, paper_);
    changed_ = true;
    anything_ = true;
    history_.push_back(current_);
    if (current_.tool == Tool::Eraser)
        EraseInk(current_);
    else
        ink_.push_back(current_);
    current_ = {};
    examStatus_.clear();
    sinceStroke_ = 0.0f;
    dirty_ = !ink_.empty();
    if (ink_.empty())
    {
        reading_ = {};              // everything was rubbed out: back to "not drawn yet"
        error_.clear();
        ++generation_;
    }
}

// The classifier gets strokes, so rubbing out has to reach them too: points under the eraser leave the strokes
// drawn before it (a stroke cut in the middle becomes two).
void DoodleBoard::EraseInk(const Stroke& eraser)
{
    const float reach = HalfWidth(Tool::Eraser, eraser.thick) + 1.0f;
    auto erased = [&](sa::Vec2 p) {
        const auto& e = eraser.points;
        for (size_t i = 0; i < e.size(); ++i)
            if (Distance(p, e[i], e[std::min(i + 1, e.size() - 1)]) <= reach)
                return true;
        return false;
    };
    std::vector<Stroke> kept;
    for (const Stroke& stroke : ink_)
    {
        Stroke piece{stroke.tool, stroke.colour, stroke.thick, {}};
        for (sa::Vec2 p : stroke.points)
        {
            if (!erased(p))
            {
                piece.points.push_back(p);
                continue;
            }
            if (!piece.points.empty())
                kept.push_back(piece);
            piece.points.clear();
        }
        if (!piece.points.empty())
            kept.push_back(std::move(piece));
    }
    ink_ = std::move(kept);
}

// --- Painting a stroke into a picture --------------------------------------------------------------------------

// Each stroke is first drawn as coverage (0..1 per pixel, the largest of its segments, so a stroke never paints
// over itself) and where across the brush each pixel lies (-1..1, for bristle streaks), then laid on the paper
// in its colour. The looks follow the shelter painter's brushes (sketch/oilprocess.py taper and flat).
void DoodleBoard::Paint(const Stroke& stroke, std::vector<std::uint8_t>& rgba)
{
    const auto& pts = stroke.points;
    if (pts.empty())
        return;
    const float width = HalfWidth(stroke.tool, stroke.thick);
    std::vector<float> half(pts.size(), width);
    if (stroke.tool == Tool::Taper)
    {
        // Full width in the middle of the stroke, pointed at both ends.
        std::vector<float> along(pts.size(), 0.0f);
        for (size_t i = 1; i < pts.size(); ++i)
        {
            const sa::Vec2 d = pts[i] - pts[i - 1];
            along[i] = along[i - 1] + std::sqrt(d.x * d.x + d.y * d.y);
        }
        const float length = along.back();
        for (size_t i = 0; i < pts.size(); ++i)
        {
            const float t = length > 1.0f ? 2.0f * along[i] / length - 1.0f : 0.0f;
            half[i] = length > 1.0f ? std::max(0.6f, width * std::pow(std::max(0.0f, 1.0f - t * t), 0.7f))
                                    : width * 0.5f;
        }
    }

    float minX = 1e9f, minY = 1e9f, maxX = -1e9f, maxY = -1e9f;
    for (sa::Vec2 p : pts)
        minX = std::min(minX, p.x), minY = std::min(minY, p.y), maxX = std::max(maxX, p.x), maxY = std::max(maxY, p.y);
    const float pad = width + 3.0f;
    const int x0 = std::clamp(static_cast<int>(std::floor(minX - pad)), 0, kResW);
    const int x1 = std::clamp(static_cast<int>(std::ceil(maxX + pad)), 0, kResW);
    const int y0 = std::clamp(static_cast<int>(std::floor(minY - pad)), 0, kResH);
    const int y1 = std::clamp(static_cast<int>(std::ceil(maxY + pad)), 0, kResH);
    if (x0 >= x1 || y0 >= y1)
        return;
    for (int y = y0; y < y1; ++y)
        std::fill_n(cover_.begin() + static_cast<std::ptrdiff_t>(y) * kResW + x0, x1 - x0, 0.0f);

    auto segment = [&](size_t i, size_t j) {
        const sa::Vec2 a = pts[i], b = pts[j];
        const sa::Vec2 ab = b - a;
        const float len = std::sqrt(ab.x * ab.x + ab.y * ab.y);
        const sa::Vec2 dir = len > 1e-4f ? sa::Vec2{ab.x / len, ab.y / len} : sa::Vec2{1.0f, 0.0f};
        const float ha = half[i], hb = half[j];
        // The flat brush's segments overlap at the joints (by its corner radius) so the rounded corners only show
        // at the two ends of the stroke.
        const float rho = 0.35f * width;
        const float ext0 = i == 0 ? 0.1f * width : rho, ext1 = j + 1 == pts.size() ? 0.1f * width : rho;
        const float r = std::max(ha, hb) + std::max(ext0, ext1) + 2.0f;
        const int sx0 = std::max(x0, static_cast<int>(std::floor(std::min(a.x, b.x) - r)));
        const int sx1 = std::min(x1, static_cast<int>(std::ceil(std::max(a.x, b.x) + r)));
        const int sy0 = std::max(y0, static_cast<int>(std::floor(std::min(a.y, b.y) - r)));
        const int sy1 = std::min(y1, static_cast<int>(std::ceil(std::max(a.y, b.y) + r)));
        for (int y = sy0; y < sy1; ++y)
            for (int x = sx0; x < sx1; ++x)
            {
                const sa::Vec2 d{x + 0.5f - a.x, y + 0.5f - a.y};
                const float u = d.x * dir.x + d.y * dir.y;
                const float v = -d.x * dir.y + d.y * dir.x;
                float c = 0.0f, phase = 0.0f;
                if (stroke.tool == Tool::Flat)
                {
                    // A rounded rectangle from -ext0 to len + ext1 along the stroke, `width` either side of it.
                    const float centre = (len + ext1 - ext0) / 2, halfLength = (len + ext0 + ext1) / 2;
                    const float qx = std::abs(u - centre) - halfLength + rho, qy = std::abs(v) - width + rho;
                    const float outside = std::sqrt(std::max(qx, 0.0f) * std::max(qx, 0.0f) +
                                                    std::max(qy, 0.0f) * std::max(qy, 0.0f));
                    const float sdf = outside + std::min(std::max(qx, qy), 0.0f) - rho;
                    c = std::clamp(0.5f - sdf, 0.0f, 1.0f);
                    phase = v / width;
                }
                else
                {
                    // Round ends; the taper's width changes along the segment.
                    const float t = len > 1e-4f ? std::clamp(u / len, 0.0f, 1.0f) : 0.0f;
                    const float h = ha + (hb - ha) * t;
                    const float du = u - t * len;
                    c = std::clamp(h + 0.5f - std::sqrt(du * du + v * v), 0.0f, 1.0f);
                    phase = v / std::max(h, 0.5f);
                }
                const size_t k = static_cast<size_t>(y) * kResW + x;
                if (c > cover_[k])
                {
                    cover_[k] = c;
                    phase_[k] = phase;
                }
            }
    };
    if (pts.size() == 1)
        segment(0, 0);
    for (size_t i = 0; i + 1 < pts.size(); ++i)
        segment(i, i + 1);

    const Pigment& paint = kPalette[static_cast<size_t>(stroke.colour)];
    const float rgb[3] = {static_cast<float>(paint.r), static_cast<float>(paint.g), static_cast<float>(paint.b)};
    const float seed = std::fmod(pts[0].x * 0.37f + pts[0].y * 0.73f, 6.2831853f);
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x)
        {
            const size_t k = static_cast<size_t>(y) * kResW + x;
            float a = cover_[k];
            if (a <= 0.0f)
                continue;
            float shade = 1.0f;
            switch (stroke.tool)
            {
            case Tool::Pencil:
                a *= 0.72f + 0.28f * Grain(x, y);
                break;
            case Tool::Flat:        // faint bristle streaks along the stroke: a little lighter or darker, a little thinner
                shade = 1.0f + 0.07f * std::sin(phase_[k] * 9.0f + seed);
                a *= 0.9f + 0.1f * std::sin(phase_[k] * 5.0f + seed * 1.7f);
                break;
            case Tool::Taper:
                shade = 1.0f + 0.06f * std::sin(phase_[k] * 9.0f + seed);
                break;
            case Tool::Eraser:
                break;
            }
            // Paint goes over what is there ("over", straight alpha); the eraser takes paint away, so the cloth (and,
            // in the reader's picture, the paper) shows again.
            std::uint8_t* dst = &rgba[k * 4];
            const float below = dst[3] / 255.0f;
            if (stroke.tool == Tool::Eraser)
            {
                dst[3] = static_cast<std::uint8_t>(std::lround(dst[3] * (1.0f - a)));
                continue;
            }
            const float out = a + below * (1.0f - a);
            for (int ch = 0; ch < 3; ++ch)
            {
                const float paint_ = std::clamp(rgb[ch] * shade, 0.0f, 255.0f);
                dst[ch] = static_cast<std::uint8_t>(std::lround((paint_ * a + dst[ch] * below * (1.0f - a)) / out));
            }
            dst[3] = static_cast<std::uint8_t>(std::lround(out * 255.0f));
        }
}

void DoodleBoard::Show(sa::Backend& backend)
{
    if (previewDirty_)
    {
        previewDirty_ = false;
        const std::vector<std::uint8_t> picture = Picture();
        if (preview_)
            backend.UpdateTexture(preview_, picture.data());
        else
            preview_ = backend.CreateTexture(kSent, kSent, picture.data(), sa::TextureFilter::Smooth);
    }
    if (!changed_)
        return;
    changed_ = false;
    shown_ = paper_;
    if (penDown_)
        Paint(current_, shown_);
    if (texture_)
        backend.UpdateTexture(texture_, shown_.data());
    else
        texture_ = backend.CreateTexture(kResW, kResH, shown_.data(), sa::TextureFilter::Smooth);
}

// --- Reading -------------------------------------------------------------------------------------------------------

// Strokes in the sent picture's pixels (0..kSent, the same square the reader gets): [xs, ys] pairs of the pencil
// strokes as they stand after any rubbing out (what read_doodle's classifier gets), or every stroke as drawn with its
// tool, paint and size.
nlohmann::json DoodleBoard::Strokes(bool pencilOnly) const
{
    nlohmann::json strokes = nlohmann::json::array();
    const float k = static_cast<float>(kSent) / kResW;
    for (const Stroke& s : pencilOnly ? ink_ : history_)
    {
        if (pencilOnly && s.tool != Tool::Pencil)
            continue;
        nlohmann::json xs = nlohmann::json::array(), ys = nlohmann::json::array();
        for (sa::Vec2 p : s.points)
        {
            xs.push_back(std::round(p.x * k));
            ys.push_back(std::round((p.y + kPadTop) * k));
        }
        if (pencilOnly)
            strokes.push_back({xs, ys});
        else
            strokes.push_back({{"tool", ToolCode(s.tool)},
                               {"colour", s.tool == Tool::Eraser ? nlohmann::json() : nlohmann::json(PaintCode(s.colour))},
                               {"thick", s.thick},
                               {"xs", xs},
                               {"ys", ys}});
    }
    return strokes;
}

// The paint laid on the beige paper, centred in a kResW square (paper above and below if it were not square), each
// sent pixel the average of 3 × 3.
std::vector<std::uint8_t> DoodleBoard::Picture() const
{
    constexpr int f = kResW / kSent;
    std::vector<std::uint8_t> image(static_cast<size_t>(kSent) * kSent * 4);
    for (int y = 0; y < kSent; ++y)
        for (int x = 0; x < kSent; ++x)
        {
            float sum[3] = {0, 0, 0};
            for (int dy = 0; dy < f; ++dy)
                for (int dx = 0; dx < f; ++dx)
                {
                    const int py = y * f + dy - kPadTop, px = x * f + dx;
                    const std::uint8_t* src = py >= 0 && py < kResH ? &paper_[(static_cast<size_t>(py) * kResW + px) * 4]
                                                                   : nullptr;
                    const float a = src ? src[3] / 255.0f : 0.0f;
                    for (int ch = 0; ch < 3; ++ch)
                        sum[ch] += (src ? src[ch] * a : 0.0f) + kPaper[ch] * (1.0f - a);
                }
            std::uint8_t* out = &image[(static_cast<size_t>(y) * kSent + x) * 4];
            for (int ch = 0; ch < 3; ++ch)
                out[ch] = static_cast<std::uint8_t>(std::lround(sum[ch] / (f * f)));
            out[3] = 255;
        }
    return image;
}

nlohmann::json DoodleBoard::Image() const
{
    return {{"width", kSent}, {"height", kSent}, {"rgba_b64", Base64(Picture())}};
}

nlohmann::json DoodleBoard::Request() const
{
    return {{"type", "read_doodle"}, {"strokes", Strokes(true)}, {"image", Image()}};
}

void DoodleBoard::SendReading()
{
    dirty_ = false;
    if (ink_.empty())
        return;
    requested_ = generation_;
    awaiting_ = Awaiting::Reading;
    client_.Send(Request());
}

void DoodleBoard::Poll()
{
    const auto reply = client_.Poll();
    if (!reply)
        return;
    const Awaiting was = std::exchange(awaiting_, Awaiting::Nothing);
    const std::string type = reply->value("type", "");
    if (was == Awaiting::Exam)
    {
        examFailed_ = type != "save_doodle_exam_result" || !reply->value("ok", false);
        const std::string path = reply->value("path", "");
        examStatus_ = examFailed_ ? "沒存成：" + reply->value("message", std::string("the server did not save it."))
                                  : "已存成考卷：" + path.substr(path.find_last_of("/\\") + 1);
        return;
    }
    if (requested_ != generation_)
        return;                     // a reading of a drawing that was cleared since
    if (type != "read_doodle_result")
    {
        error_ = reply->value("message", "The server could not read the doodle.");
        return;
    }
    error_.clear();
    reading_.have = true;
    reading_.isCat = reply->value("is_cat", false);
    reading_.top.clear();
    // The first two guesses with their probabilities, so a hesitation between two classes shows.
    if (const auto& top = (*reply)["top_k"]; top.is_array())
        for (size_t i = 0; i < std::min<size_t>(2, top.size()); ++i)
            if (top[i].is_object())
            {
                char p[16];
                std::snprintf(p, sizeof p, "%.2f", top[i].value("p", 0.0));
                reading_.top += (reading_.top.empty() ? "" : "／") + top[i].value("label", std::string("?")) + " " + p;
            }
    reading_.attributes = reply->value("attributes", nlohmann::json::object());
    reading_.reader = reply->value("reader", "none");
}

// --- The exam form ---------------------------------------------------------------------------------------------

namespace
{
constexpr sa::Vec2 kFormPos{70, 24}, kFormSize{1140, 672};
constexpr sa::Vec2 kPreviewPos{140, 150};
constexpr float kPreviewSize = 300.0f;
constexpr float kFieldLabelX = 470.0f, kFieldX = 650.0f, kFieldW = 250.0f, kFieldH = 34.0f;
constexpr float kFieldY = 132.0f, kFieldStep = 42.0f, kOptionH = 32.0f;
constexpr sa::Vec2 kOkPos{650, 616}, kOkSize{220, 50}, kCancelPos{890, 616}, kCancelSize{170, 50};

sa::Vec2 FieldPos(int i) { return {kFieldX, kFieldY + i * kFieldStep}; }

// Where field i's open list is: below it, or above it when it would run off the form.
sa::Vec2 ListPos(int i, size_t options)
{
    const sa::Vec2 f = FieldPos(i);
    const float h = options * kOptionH;
    return {f.x, f.y + kFieldH + h <= kFormPos.y + kFormSize.y - 8 ? f.y + kFieldH : f.y - h};
}
} // namespace

// Every field starts at what the reader says now (unspecified while there is no reader); the owner changes what is
// wrong. A name the reader gives that the form does not list is added to that field's list, so the prefill is exact.
void DoodleBoard::OpenForm()
{
    auto field = [&](std::string label, std::string key, bool patch, std::vector<Option> options) {
        options.insert(options.begin(), {"", kUnspecified});
        const nlohmann::json& a = reading_.attributes;
        const auto said = patch ? Field(a.is_object() && a.contains("patches") ? a["patches"] : nlohmann::json(), key.c_str())
                                : Field(a, key.c_str());
        int prefill = 0;
        if (said)
        {
            auto it = std::find_if(options.begin(), options.end(), [&](const Option& o) { return o.code == *said; });
            if (it == options.end())
                it = options.insert(options.end(), {*said, *said});
            prefill = static_cast<int>(it - options.begin());
        }
        form_.fields.push_back({std::move(label), std::move(key), patch, std::move(options), prefill, prefill});
    };
    auto names = [](const auto& list, bool withNone) {
        std::vector<Option> out;
        if (withNone)
            out.push_back({"none", "沒有斑"});
        for (const Name& n : list)
            if (!withNone || std::string(n.code) != "mixed")
                out.push_back({n.code, n.zh});
        return out;
    };
    form_ = {};
    form_.open = true;
    previewDirty_ = true;           // the form shows the picture the reader got
    field("整體毛色", "coat", false, names(kCoats, false));
    field("花紋", "pattern", false, names(kPatterns, false));
    for (const auto& [code, zh] : kRegions)
        field(std::string("斑塊：") + zh, code, true, names(kCoats, true));
    std::vector<Option> eyes;
    for (const Name& n : kEyes)
        if (std::string(n.code) != "copper")    // the exam's eye colours (the reader may still say copper: then it is added)
            eyes.push_back({n.code, n.zh});
    field("左眼", "left_eye", false, eyes);
    field("右眼", "right_eye", false, eyes);
}

void DoodleBoard::UpdateForm(sa::Backend& backend, const ui::Mouse& mouse)
{
    if (backend.KeyPressed(sa::Key::Escape))
    {
        if (form_.dropdown >= 0)
            form_.dropdown = -1;
        else
            form_.open = false;
        return;
    }
    if (!mouse.JustPressed())
        return;
    const sa::Vec2 m = mouse.Position();
    if (form_.dropdown >= 0)
    {
        FormField& f = form_.fields[static_cast<size_t>(form_.dropdown)];
        const sa::Vec2 list = ListPos(form_.dropdown, f.options.size());
        if (Inside(m, list, {kFieldW, f.options.size() * kOptionH}))
            f.choice = std::clamp(static_cast<int>((m.y - list.y) / kOptionH), 0, static_cast<int>(f.options.size()) - 1);
        form_.dropdown = -1;        // any click closes the list
        return;
    }
    for (int i = 0; i < static_cast<int>(form_.fields.size()); ++i)
        if (Inside(m, FieldPos(i), {kFieldW, kFieldH}))
            form_.dropdown = i;
    if (Inside(m, kCancelPos, kCancelSize))
        form_.open = false;
    else if (Inside(m, kOkPos, kOkSize))
    {
        // The answer, and which fields differ from the reader's prefill.
        nlohmann::json answer = {{"patches", nlohmann::json::object()}}, edited = {{"patches", nlohmann::json::object()}};
        for (const FormField& f : form_.fields)
        {
            const std::string& code = f.options[static_cast<size_t>(f.choice)].code;
            nlohmann::json& a = f.patch ? answer["patches"] : answer;
            nlohmann::json& e = f.patch ? edited["patches"] : edited;
            a[f.key] = code.empty() ? nlohmann::json() : nlohmann::json(code);
            e[f.key] = f.choice != f.prefill;
        }
        exam_ = {{"type", "save_doodle_exam"}, {"image", Image()}, {"strokes", Strokes(false)},
                 {"pencil_strokes", Strokes(true)}, {"answer", answer}, {"edited", edited},
                 {"game_version", SA_GAME_VERSION}};
        form_.open = false;
        examQueued_ = true;
        examFailed_ = false;
        examStatus_ = "存檔中…";
    }
}

void DoodleBoard::SendExam()
{
    examQueued_ = false;
    awaiting_ = Awaiting::Exam;
    client_.Send(std::move(exam_));
    exam_ = {};
}

void DoodleBoard::DrawForm(sa::Backend& b, sa::Vec2 mouse) const
{
    if (!form_.open)
        return;
    b.DrawRect({0, 0}, {kScreenW, 720}, {0.22f, 0.13f, 0.10f, 0.45f});
    ui::Panel(b, kFormPos, kFormSize);
    sa::DrawText(b, {150, 62}, "存下塗鴉當考卷：確認這隻貓的真實答案", 2.3f, ui::kInk);
    sa::DrawText(b, {150, 100}, "已先填入讀取模型現在讀到的；不對的就改，沒畫到的選「未指定」。左右以畫面為準。", 1.5f,
                 ui::kMuted);
    if (preview_)
        b.DrawSprite(preview_, {}, kPreviewPos, {kPreviewSize, kPreviewSize});
    Outline(b, kPreviewPos, {kPreviewSize, kPreviewSize}, 1, ui::kBorder);
    sa::DrawWrapped(b, {kPreviewPos.x, kPreviewPos.y + kPreviewSize + 14},
                    "讀取模型：" + (reading_.reader == "none" ? std::string("還沒裝好（預填都是未指定）") : reading_.reader),
                    kPreviewSize, 1.5f, ui::kMuted);
    sa::DrawText(b, {kPreviewPos.x, kPreviewPos.y + kPreviewSize + 60}, "橘框＝改過", 1.5f, ui::kAccent);

    for (int i = 0; i < static_cast<int>(form_.fields.size()); ++i)
    {
        const FormField& f = form_.fields[static_cast<size_t>(i)];
        const sa::Vec2 pos = FieldPos(i);
        sa::DrawText(b, {kFieldLabelX, pos.y + 6}, f.label, 1.8f, ui::kInk);
        const bool edited = f.choice != f.prefill;
        b.DrawRect(pos, {kFieldW, kFieldH}, Inside(mouse, pos, {kFieldW, kFieldH}) ? sa::Color{1.0f, 0.95f, 0.90f, 1.0f}
                                                                                    : sa::Color{1.0f, 0.99f, 0.97f, 1.0f});
        Outline(b, pos, {kFieldW, kFieldH}, edited ? 2.0f : 1.0f, edited ? ui::kAccent : ui::kBorder);
        const Option& o = f.options[static_cast<size_t>(f.choice)];
        sa::DrawText(b, {pos.x + 12, pos.y + 7}, o.zh, 1.7f, o.code.empty() ? ui::kMuted : ui::kInk);
        sa::DrawText(b, {pos.x + kFieldW - 24, pos.y + 7}, "v", 1.6f, ui::kMuted);
    }
    ui::DrawButton(b, {kOkPos, kOkSize, "確定存下", true, ui::ButtonStyle::Green}, mouse, 1.9f);
    ui::DrawButton(b, {kCancelPos, kCancelSize, "取消", true, ui::ButtonStyle::Cream}, mouse, 1.9f);

    if (form_.dropdown >= 0)
    {
        const FormField& f = form_.fields[static_cast<size_t>(form_.dropdown)];
        const sa::Vec2 list = ListPos(form_.dropdown, f.options.size());
        const sa::Vec2 size{kFieldW, f.options.size() * kOptionH};
        b.DrawRect({list.x + 4, list.y + 5}, size, {0.25f, 0.16f, 0.10f, 0.25f});
        b.DrawRect(list, size, {1.0f, 0.99f, 0.97f, 1.0f});
        Outline(b, list, size, 1, ui::kBorder);
        for (size_t k = 0; k < f.options.size(); ++k)
        {
            const sa::Vec2 row{list.x, list.y + k * kOptionH};
            if (static_cast<int>(k) == f.choice)
                b.DrawRect(row, {kFieldW, kOptionH}, {0.99f, 0.86f, 0.89f, 1.0f});
            else if (Inside(mouse, row, {kFieldW, kOptionH}))
                b.DrawRect(row, {kFieldW, kOptionH}, {1.0f, 0.95f, 0.90f, 1.0f});
            const Option& o = f.options[k];
            sa::DrawText(b, {row.x + 12, row.y + 6}, o.zh + (static_cast<int>(k) == f.prefill ? "（模型）" : ""), 1.7f,
                         o.code.empty() ? ui::kMuted : ui::kInk);
        }
    }
}

// --- Drawing -------------------------------------------------------------------------------------------------------

// The stall again, its cloth left for Brush GPT: the same canvas the game's drawings are guessed on.
void DoodleBoard::DrawFrame(sa::Backend& b) const
{
    const float cx = stage::ToScreen({stage::kFrameX, 0}).x;
    if (cx < -kScreenW / 2 || cx > kScreenW * 1.5f)
        return;
    stage::DrawStall(b, stage::kFrameX);
    const sa::Vec2 cloth = stage::ClothScreenPosAt(stage::kFrameX), size = stage::ClothSize();
    const float scale = 1.8f;
    const auto lines = sa::WrapText("之後 Brush GPT\n會依左邊的描述\n在這裡畫一隻油畫貓", size.x - 24, scale);
    float y = cloth.y + (size.y - lines.size() * sa::LineHeight(scale)) / 2;
    for (const auto& line : lines)
    {
        sa::DrawText(b, {cloth.x + (size.x - sa::TextWidth(line, scale)) / 2, y}, line, scale, ui::kMuted);
        y += sa::LineHeight(scale);
    }
}

// The stall copy with the paint on its cloth, the tools and the reading panel (in front of the girl, so she never
// hides the drawing).
void DoodleBoard::Draw(sa::Backend& b, sa::Vec2 mouse) const
{
    const sa::Vec2 o = Origin();
    if (o.x < -kScreenW || o.x > kScreenW)
        return;
    stage::DrawStall(b, StallWorldX());
    if (texture_)
        b.DrawSprite(texture_, {}, ClothPos(), stage::ClothSize());
    DrawTools(b, o, mouse);
    DrawPanel(b, o, mouse);
}

void DoodleBoard::DrawTools(sa::Backend& b, sa::Vec2 o, sa::Vec2 mouse) const
{
    const sa::Vec2 pos{o.x + kToolsPos.x, kToolsPos.y};
    if (!stage::DrawRounded(b, pos, kToolsSize, ui::kPanel, ui::kBorder))
        b.DrawRect(pos, kToolsSize, ui::kPanel);
    const float label = 1.6f;
    const auto at = [&](sa::Vec2 p) { return sa::Vec2{o.x + p.x, p.y}; };
    sa::DrawText(b, at({kToolX, kSwatchY - 24}), "顏色", label, ui::kInk);
    for (int i = 0; i < static_cast<int>(kPalette.size()); ++i)
    {
        const Pigment& p = kPalette[static_cast<size_t>(i)];
        const sa::Vec2 s = at(SwatchPos(i));
        if (i == colour_ && tool_ != Tool::Eraser)
            Outline(b, {s.x - 3, s.y - 3}, {kSwatchW + 6, kSwatchH + 6}, 3, ui::kAccent);
        Swatch(b, s, {kSwatchW, kSwatchH}, Rgb(p.r, p.g, p.b));
    }

    auto tile = [&](sa::Vec2 p, sa::Vec2 size, bool selected) {
        const sa::Vec2 s = at(p);
        const bool hover = Inside(mouse, s, size);
        b.DrawRect(s, size, selected ? sa::Color{0.99f, 0.86f, 0.89f, 1.0f}
                                     : hover ? sa::Color{1.0f, 0.95f, 0.90f, 1.0f} : sa::Color{1.0f, 0.99f, 0.97f, 1.0f});
        Outline(b, s, size, selected ? 2.0f : 1.0f, selected ? ui::kAccent : ui::kBorder);
        return s;
    };
    sa::DrawText(b, at({kToolX, kPenY - 24}), "筆", label, ui::kInk);
    for (int i = 0; i < 4; ++i)
    {
        const Tool t = static_cast<Tool>(i);
        const sa::Vec2 s = tile(PenPos(i), {kPenW, kPenH}, tool_ == t);
        const float y = s.y + 15;
        switch (t)
        {
        case Tool::Pencil:
            b.DrawLine({s.x + 14, y + 4}, {s.x + kPenW - 14, y - 4}, 2, ui::kInk);
            break;
        case Tool::Flat:
            b.DrawRect({s.x + 16, y - 6}, {kPenW - 32, 12}, ui::kInk);
            for (int k = 0; k < 3; ++k)
                b.DrawRect({s.x + 16, y - 4 + k * 4}, {kPenW - 32, 1}, ui::kMuted);
            break;
        case Tool::Taper:
            for (int k = 0; k < 12; ++k)
            {
                const float t0 = k / 12.0f, t1 = (k + 1) / 12.0f, tm = (t0 + t1) - 1.0f;
                const float w = 1.0f + 7.0f * std::pow(std::max(0.0f, 1.0f - tm * tm), 0.7f);
                b.DrawLine({s.x + 12 + t0 * (kPenW - 24), y}, {s.x + 12 + t1 * (kPenW - 24), y}, w, ui::kInk);
            }
            break;
        case Tool::Eraser:
            b.DrawRect({s.x + 24, y - 7}, {kPenW - 48, 14}, ui::kButton);
            b.DrawRect({s.x + 24, y - 7}, {10, 14}, ui::kButtonHover);
            break;
        }
        const char* name = ToolName(t);
        sa::DrawText(b, {s.x + (kPenW - sa::TextWidth(name, 1.45f)) / 2, s.y + 29}, name, 1.45f, ui::kInk);
    }
    sa::DrawText(b, at({kToolX, kSizeY - 24}), "粗細", label, ui::kInk);
    for (int i = 0; i < 2; ++i)
    {
        const sa::Vec2 s = tile(SizePos(i), {kPenW, kSizeH}, thick_ == (i == 1));
        b.DrawLine({s.x + 12, s.y + kSizeH / 2}, {s.x + 40, s.y + kSizeH / 2}, i == 1 ? 8.0f : 2.0f, ui::kInk);
        sa::DrawText(b, {s.x + 50, s.y + 8}, i == 1 ? "粗" : "細", 1.6f, ui::kInk);
    }
    ui::DrawButton(b, {at(kClearPos), kClearSize, "清除", anything_, ui::ButtonStyle::Cream}, mouse, 1.8f);
}

void DoodleBoard::DrawPanel(sa::Backend& b, sa::Vec2 o, sa::Vec2 mouse) const
{
    const sa::Vec2 pos{o.x + kPanelPos.x, kPanelPos.y};
    ui::Panel(b, pos, kPanelSize);
    const sa::Vec2 in = ui::PanelInset(b, kPanelSize);
    const float x = pos.x + in.x, right = pos.x + kPanelSize.x - in.x;
    float y = pos.y + std::max(in.y, 24.0f);
    sa::DrawText(b, {x, y}, "AI 讀到的貓", 2.4f, ui::kInk);
    const char* side = "左右以畫面為準";
    sa::DrawText(b, {right - sa::TextWidth(side, 1.5f), y + 6}, side, 1.5f, ui::kMuted);
    y += 48;

    const float scale = 1.8f, valueX = x + 96, rowH = 34;
    const bool drawn = !ink_.empty();
    const nlohmann::json none = nlohmann::json::object();
    const nlohmann::json& attributes = drawn && reading_.have ? reading_.attributes : none;
    auto label = [&](const char* text) { sa::DrawText(b, {x, y}, text, scale, ui::kMuted); };
    auto unspecified = [&](float at) { sa::DrawText(b, {at, y}, kUnspecified, scale, ui::kMuted); };
    // A name the reader gave, in Chinese with its colour beside it (a name the game does not know is shown as is).
    auto named = [&](const std::optional<std::string>& code, const Name* known) {
        if (!code)
            return unspecified(valueX);
        float at = valueX;
        if (known && (known->r || known->g || known->b))
        {
            Swatch(b, {at, y + 1}, {16, 16}, Rgb(known->r, known->g, known->b));
            at += 24;
        }
        sa::DrawText(b, {at, y}, known ? known->zh : *code, scale, ui::kInk);
    };

    label("是不是貓");
    if (!drawn)
        sa::DrawText(b, {valueX, y}, "（還沒畫）", scale, ui::kMuted);
    else if (!HasPencil())          // the classifier only reads the pencil outline
        sa::DrawText(b, {valueX, y}, "請先用鉛筆畫出輪廓", scale, ui::kMuted);
    else if (!reading_.have)
        sa::DrawText(b, {valueX, y}, "讀取中…", scale, ui::kMuted);
    else
    {
        sa::DrawText(b, {valueX, y}, reading_.isCat ? "是" : "不是", scale, reading_.isCat ? ui::kGood : ui::kBad);
        if (!reading_.top.empty())          // the first two guesses, on a line of their own
        {
            y += 24;
            sa::DrawText(b, {valueX, y}, "前兩名：" + reading_.top, 1.5f, ui::kMuted);
        }
    }
    y += rowH;

    label("整體顏色");
    const auto coat = Field(attributes, "coat");
    named(coat, coat ? Find(kCoats, *coat) : nullptr);
    y += rowH;

    label("花紋");
    const auto pattern = Field(attributes, "pattern");
    named(pattern, pattern ? Find(kPatterns, *pattern) : nullptr);
    if (pattern)
    {
        const char* stripes = *pattern == "tabby" ? "（有虎紋）" : "（沒有虎紋）";
        const Name* known = Find(kPatterns, *pattern);
        sa::DrawText(b, {valueX + sa::TextWidth(known ? known->zh : *pattern, scale) + 8, y}, stripes, scale,
                     ui::kMuted);
    }
    y += rowH;

    label("斑塊");
    {
        const nlohmann::json patches = attributes.is_object() && attributes.contains("patches") &&
                                               attributes["patches"].is_object()
                                           ? attributes["patches"]
                                           : none;
        std::string list;
        bool noPatch = false;       // the reader said "none" somewhere: it looked and found no patch there
        for (const auto& [code, zh] : kRegions)
        {
            const auto colour = Field(patches, code);
            if (!colour)
                continue;
            if (*colour == "none")
            {
                noPatch = true;
                continue;
            }
            const Name* known = Find(kCoats, *colour);
            list += (list.empty() ? "" : "、") + std::string(zh) + " " + (known ? known->zh : *colour);
        }
        if (list.empty())
            sa::DrawText(b, {valueX, y}, noPatch ? "沒有斑塊" : kUnspecified, scale, noPatch ? ui::kInk : ui::kMuted);
        else
            y += sa::DrawWrapped(b, {valueX, y}, list, right - valueX, scale, ui::kInk) - sa::LineHeight(scale);
    }
    y += rowH;

    label("左眼");
    const auto left = Field(attributes, "left_eye");
    named(left, left ? Find(kEyes, *left) : nullptr);
    y += rowH;
    label("右眼");
    const auto rightEye = Field(attributes, "right_eye");
    named(rightEye, rightEye ? Find(kEyes, *rightEye) : nullptr);
    y += rowH + 6;

    b.DrawRect({x, y}, {right - x, 1}, ui::kBorder);
    y += 14;
    const float note = 1.5f;
    std::string about = "毛色、花紋、斑塊、眼睛由讀取模型給；它沒說的就是未指定。";
    if (drawn && reading_.have)
        about = reading_.reader == "none" ? "讀取模型還沒裝好，所以毛色、花紋、斑塊、眼睛現在都是未指定。"
                                          : "讀取模型：" + reading_.reader;
    y += sa::DrawWrapped(b, {x, y}, about, right - x, note, ui::kMuted);
    if (!error_.empty())
        sa::DrawWrapped(b, {x, y + 4}, error_, right - x, note, ui::kBad);
    else if (drawn && (dirty_ || awaiting_ == Awaiting::Reading))
        sa::DrawText(b, {x, y + 4}, "更新中…", note, ui::kAccent);

    // Keeping the doodle as an exam: only once the reading is for the drawing as it stands, so the form's prefill
    // is the reader's answer for exactly this picture.
    const bool canSave = drawn && Current() && !examQueued_ && awaiting_ != Awaiting::Exam;
    ui::DrawButton(b, {{o.x + kSavePos.x, kSavePos.y}, kSaveSize, "存下塗鴉（當考卷）", canSave, ui::ButtonStyle::Pink},
                   mouse, 1.7f);
    if (!examStatus_.empty())
        sa::DrawText(b, {o.x + kSavePos.x, kSavePos.y - 24}, examStatus_, note, examFailed_ ? ui::kBad : ui::kGood);
}
