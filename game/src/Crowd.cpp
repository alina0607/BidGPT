#include "Crowd.h"

#include "Scenes.h"
#include "Stage.h"

#include "sa/Text.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iostream>

namespace
{

constexpr float kCell = 64.0f;         // a buyer sprite cell, drawn at stage::BuyerScale()
constexpr float kTagW = 128.0f;
constexpr float kTagH = 52.0f;
constexpr float kRowHalf = 330.0f;     // the row spans the stall's front: world x -330..330
constexpr float kMaxGap = 165.0f;
constexpr float kInFront = 80.0f;      // stop deep in the foreground flowers, well clear of the cloth

float Sprite() { return kCell * stage::BuyerScale(); }
constexpr float kNoteW = 230.0f;       // compact enough for alternating bubbles not to overlap
constexpr float kNoteScale = 1.45f;

// Tags sit under the buyers' feet, so they never cover the drawing on the cloth; notes go over their heads.
float TagTop(const CrowdSpot& spot) { return spot.feet.y + 4; }
float HeadTop(const CrowdSpot& spot) { return spot.feet.y - std::max(Sprite(), stage::NpcHeight()) + 6; }

enum class NotePlacement
{
    Above,
    Left,
    Right,
};

// A speech note whose bottom edge is at `bottom`, sized to its text plus the room its frame and flowers
// take. Outer buyers grow toward the screen edge; only the inner buyers use the bubble above their heads.
void Note(sa::Backend& b, float x, float bottom, const std::vector<std::pair<std::string, sa::Color>>& lines,
          bool question = false, NotePlacement placement = NotePlacement::Above)
{
    float textW = 0;
    for (const auto& [text, color] : lines)
        textW = std::max(textW, sa::TextWidth(text, kNoteScale));
    const float textH = static_cast<float>(lines.size()) * sa::LineHeight(kNoteScale);
    const sa::Vec2 size = stage::UiSkinFit(b, stage::UiSkin::Speech, {textW, textH});
    const auto in = stage::UiSkinInsets(b, stage::UiSkin::Speech, size);
    const float wanted = placement == NotePlacement::Left    ? x - size.x - 16
                         : placement == NotePlacement::Right ? x + 16
                                                             : x - size.x / 2;
    const float left = std::clamp(wanted, 12.0f, kWidth - 12 - size.x);
    const float top = bottom - size.y;
    ui::SpeechPanel(b, {left, top}, size, question);
    float y = top + in.top;
    for (const auto& [text, color] : lines)
    {
        sa::DrawText(b, {left + in.left, y}, text, kNoteScale, color);
        y += sa::LineHeight(kNoteScale);
    }
}

void Tag(sa::Backend& b, const nlohmann::json& buyer, const CrowdSpot& spot, bool hovered)
{
    const bool present = buyer["present"].get<bool>();
    // The tag grows to its name and offer, with room for its frame and flowers.
    const std::string label = buyer["name"];
    const std::string model = ModelLabel(buyer);
    const float nameW = sa::TextWidth(label, 1.8f) + (model.empty() ? 0 : sa::TextWidth(" " + model, 1.3f));
    const float textW = std::max(nameW, sa::TextWidth("$000 -$000", 1.8f));
    const sa::Vec2 size = ui::PanelFit(b, {textW, kTagH - 14});
    const sa::Vec2 pos{spot.feet.x - size.x / 2, TagTop(spot)};
    const float inTop = ui::PanelInset(b, size).y;
    ui::Panel(b, pos, size, !present ? ui::kButtonOff : hovered ? ui::kWhite : ui::kPanel);
    const std::string name = buyer["name"];
    const float nameX = spot.feet.x - nameW / 2;
    sa::DrawText(b, {nameX, pos.y + inTop - 6}, name, 1.8f, present ? ui::kInk : ui::kMuted);
    if (!model.empty())
        sa::DrawText(b, {nameX + sa::TextWidth(name + " ", 1.8f), pos.y + inTop - 3}, model, 1.3f, ui::kMuted);
    std::string offer = !present ? "left" : buyer["offer"].is_null() ? "-" : Money(buyer["offer"]);
    std::string change;
    const nlohmann::json moved = buyer.value("change", nlohmann::json());
    if (present && moved.is_object() && std::abs(moved.value("amount", 0.0)) >= 0.5)
    {
        const double amount = moved["amount"];
        change = (amount > 0 ? " +" : " -") + Money(std::abs(amount), false);
    }
    const float w = sa::TextWidth(offer, 1.8f) + sa::TextWidth(change, 1.4f);
    const float x = spot.feet.x - w / 2;
    sa::DrawText(b, {x, pos.y + inTop + 17}, offer, 1.8f, present ? ui::kGood : ui::kMuted);
    if (!change.empty())
        sa::DrawText(b, {x + sa::TextWidth(offer, 1.8f), pos.y + inTop + 20}, change, 1.4f,
                     change[1] == '+' ? ui::kGood : ui::kBad);
}

// What the buyer cares about and what they made of the last answer.
std::vector<std::pair<std::string, sa::Color>> Details(const nlohmann::json& buyer, bool scenes)
{
    std::vector<std::pair<std::string, sa::Color>> lines;
    std::string traits;
    for (const auto& t : buyer.value("traits", nlohmann::json::array()))
        traits += (traits.empty() ? "" : ", ") + t.get<std::string>();
    for (const std::string& line : WrapLines(traits, kNoteW - 32, 4, kNoteScale))
        lines.emplace_back(line, ui::kInk);
    const nlohmann::json moved = buyer.value("change", nlohmann::json());
    if (moved.is_object() && moved.contains("reaction"))
    {
        const bool liked = moved.value("liked", false);
        for (const std::string& line : WrapLines((liked ? "+ " : "- ") + moved["reaction"].get<std::string>(),
                                                 kNoteW - 32, 2, kNoteScale))
            lines.emplace_back(line, liked ? ui::kGood : ui::kBad);
    }
    if (scenes && buyer["present"].get<bool>())
        lines.emplace_back("Click: how they picture it at home", ui::kMuted);
    return lines;
}

} // namespace

std::string ModelLabel(const nlohmann::json& buyer)
{
    const auto model = buyer.value("model", nlohmann::json());
    if (!model.is_string())
        return "";
    std::string label = model.get<std::string>();
    if (const size_t cut = label.rfind("buyer_gpt_", 0) == 0 ? 10 : 0; cut)
        label.erase(0, cut);
    if (!label.empty())
        label[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(label[0])));
    return label;
}

float DrawBuyerCards(sa::Backend& b, const nlohmann::json& buyers, sa::Vec2 pos, float width, float bottom)
{
    constexpr float kHead = 1.6f, kBody = 1.3f, kGap = 8.0f;
    std::vector<std::pair<std::string, sa::Color>> lines;
    std::vector<float> scales;
    for (const auto& buyer : buyers)
    {
        if (!buyer.value("model", nlohmann::json()).is_string())
            continue;                       // the player
        const std::string model = ModelLabel(buyer);
        lines.emplace_back(buyer["name"].get<std::string>() + (model.empty() ? "" : "  (" + model + ")"),
                           buyer["present"].get<bool>() ? ui::kInk : ui::kMuted);
        scales.push_back(kHead);
        std::string persona = buyer.value("persona", "");
        if (const size_t who = persona.find(", who "); who != std::string::npos)
            persona = persona.substr(who + 6);          // "Sam, who is retired and home all day." -> "is retired ..."
        std::string cares;
        for (const auto& t : buyer.value("traits", nlohmann::json::array()))
            cares += (cares.empty() ? "cares about: " : ", ") + t.get<std::string>();
        for (const std::string& text : {persona, cares})
            for (std::string& line : WrapLines(text, width - 90, 3, kBody))
            {
                lines.emplace_back(std::move(line), text == cares ? ui::kMuted : ui::kInk);
                scales.push_back(kBody);
            }
    }
    if (lines.empty())
        return pos.y;
    float textH = 0;
    for (size_t i = 0; i < lines.size(); ++i)
        textH += sa::LineHeight(scales[i]) + (scales[i] == kHead && i ? kGap : 0);
    const sa::Vec2 size = ui::PanelFit(b, {width - 90, std::min(textH, bottom - pos.y - 60)});
    const sa::Vec2 inset = ui::PanelInset(b, size);
    ui::Panel(b, pos, size);
    float y = pos.y + inset.y;
    for (size_t i = 0; i < lines.size(); ++i)
    {
        if (scales[i] == kHead && i)
            y += kGap;
        if (y + sa::LineHeight(scales[i]) > pos.y + size.y - inset.y + 4)
            break;                          // what does not fit is left out, never drawn over the frame
        sa::DrawText(b, {pos.x + inset.x, y}, lines[i].first, scales[i], lines[i].second);
        y += sa::LineHeight(scales[i]);
    }
    return pos.y + size.y;
}

std::string Money(double v, bool cents)
{
    char buf[32];
    if (cents && v < 10)
        std::snprintf(buf, sizeof buf, "$%.2f", v);
    else
        std::snprintf(buf, sizeof buf, "$%.0f", v);
    std::string s = buf;
    // thousands separators for the integer part
    size_t dot = s.find('.');
    size_t end = dot == std::string::npos ? s.size() : dot;
    for (int i = static_cast<int>(end) - 3; i > 1; i -= 3)
        s.insert(static_cast<size_t>(i), ",");
    return s;
}

std::vector<std::string> WrapLines(const std::string& text, float width, size_t maxLines, float scale)
{
    if (maxLines == 0)
        return {};
    auto lines = sa::WrapText(text, width, scale);
    if (lines.size() <= maxLines)
        return lines;
    lines.resize(maxLines);
    std::string& last = lines.back();
    // Drop whole words until "..." fits, so the line never ends inside a word ("hiki...").
    while (!last.empty() && sa::TextWidth(last + "...", scale) > width)
    {
        const size_t space = last.find_last_of(' ');
        last.resize(space == std::string::npos ? 0 : space);
    }
    last += "...";
    return lines;
}

std::vector<CrowdSpot> CrowdSpots(const nlohmann::json& buyers)
{
    std::vector<CrowdSpot> spots;
    const size_t n = buyers.size();
    const float gap = n > 1 ? std::min(kMaxGap, 2 * kRowHalf / static_cast<float>(n - 1)) : 0.0f;
    float x = -gap * static_cast<float>(n > 0 ? n - 1 : 0) / 2;
    for (const auto& buyer : buyers)
    {
        CrowdSpot spot{buyer["name"], stage::ToScreen({x, stage::Ground() + kInFront}), {}};
        const float top = HeadTop(spot);
        spot.hit = {{spot.feet.x - kTagW / 2, top}, {kTagW, TagTop(spot) + kTagH - top}, buyer["name"], true};
        spots.push_back(spot);
        x += gap;
    }
    return spots;
}

void DrawCrowd(sa::Backend& b, const CrowdView& view)
{
    const auto spots = CrowdSpots(*view.buyers);
    for (size_t i = 0; i < spots.size(); ++i)
    {
        if (view.walkers && view.walkers->count(spots[i].name))
            continue;   // drawn after the others: they walk in from nearer the camera
        const Look* look = view.art->LookFor(spots[i].name);
        const auto sprite = view.sprites->find(spots[i].name);
        if (look && sprite != view.sprites->end())
            sprite->second.Draw(b, *look, {spots[i].feet.x - Sprite() / 2, spots[i].feet.y - Sprite() + 6},
                                {Sprite(), Sprite()});
    }
    if (view.walkers)
        for (const CrowdSpot& spot : spots)
            if (const auto w = view.walkers->find(spot.name); w != view.walkers->end())
                w->second.Draw(b, spot.feet);
    const CrowdSpot* hovered = nullptr;
    for (size_t i = 0; i < spots.size(); ++i)
    {
        const bool over = spots[i].hit.Contains(view.mouse);
        if (over)
            hovered = &spots[i];
        const Walker* walker = nullptr;
        if (view.walkers)
            if (const auto w = view.walkers->find(spots[i].name); w != view.walkers->end())
                walker = &w->second;
        if (!walker || walker->Settled())
            Tag(b, (*view.buyers)[i], spots[i], over);
    }
    for (size_t i = 0; i < spots.size(); ++i)
    {
        const NotePlacement placement = i == 0                 ? NotePlacement::Left
                                        : i + 1 == spots.size() ? NotePlacement::Right
                                                                : NotePlacement::Above;
        const float noteBottom = placement == NotePlacement::Above
                                     ? HeadTop(spots[i]) - 8
                                     : spots[i].feet.y - stage::NpcHeight() * 0.35f;
        if (hovered == &spots[i])
            Note(b, spots[i].feet.x, noteBottom, Details((*view.buyers)[i], view.scenes), false, placement);
        else if (view.speech)
        {
            const auto said = view.speech->find(spots[i].name);
            if (said == view.speech->end() || said->second.empty())
                continue;
            std::vector<std::pair<std::string, sa::Color>> lines;
            for (const std::string& line : WrapLines(said->second, kNoteW - 32, view.speechLines, kNoteScale))
                lines.emplace_back(line, ui::kInk);
            Note(b, spots[i].feet.x, noteBottom, lines, spots[i].name == view.asker, placement);
        }
    }
}

void UpdateCrowdSprites(sa::Backend& b, CharacterArt& art, const nlohmann::json& buyers,
                        std::map<std::string, BuyerSprite>& sprites, std::map<std::string, Walker>& walkers, float dt,
                        bool autoplay)
{
    npcs::Load(b);
    if (!art.Loaded())
    {
        art.Load(b, AssetDir() + "/characters");
        if (autoplay)
            std::cout << "art: " << art.LookCount() << " looks\n";
    }
    size_t index = 0;
    for (const auto& buyer : buyers)
    {
        const std::string name = buyer["name"];
        if (index < npcs::Count())
        {
            auto walker = walkers.try_emplace(name, index, 0.7f * static_cast<float>(index)).first;
            walker->second.Update(dt, buyer["present"].get<bool>());
        }
        else if (const Look* look = art.LookFor(name))
        {
            BuyerSprite& sprite = sprites[name];
            sprite.SetMood(buyer.value("mood", "idle"), *look);
            sprite.Update(dt, *look);
        }
        ++index;
    }
}
