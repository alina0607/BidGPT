#include "Crowd.h"
#include "Scenes.h"
#include "Stage.h"
#include "Ui.h"

#include "sa/Text.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace
{

// Floating panels over the market: the conversation (or a buyer's scene) on the left, the answer or the
// sale on the right; the buyers stand in front of the counter between them (Crowd.h).
constexpr float kLogX = 20.0f;
constexpr float kLogW = 300.0f;
constexpr float kLogTop = 78.0f;
constexpr float kLogBottom = 440.0f;
constexpr float kSceneW = 920.0f;           // a buyer's scene is wider, over the stall, while it is open
constexpr float kSceneBottom = 470.0f;
constexpr float kSideX = 960.0f;
constexpr float kSideW = 300.0f;
constexpr float kSideTop = 78.0f;
constexpr float kSidePad = 14.0f;
constexpr float kSideHeadTop = 40.0f;   // below the panel's top frame and flowers (panel.png cut, scale 1)
constexpr float kSideHeadLeft = 62.0f;  // right of its left frame and flowers
constexpr float kFieldH = 36.0f;
constexpr float kFieldPad = 12.0f;
constexpr float kFieldScale = 1.8f;
constexpr float kButtonH = 40.0f;
constexpr float kButtonGap = 8.0f;
constexpr size_t kMaxTyped = 120;           // the server keeps at most this many characters too
constexpr int kScenePointsPerFrame = 4; // how fast a scene is drawn, stroke by stroke

class RoundScene final : public sa::Scene
{
public:
    RoundScene(GameContext& ctx, nlohmann::json first) : ctx_(ctx), rng_(ctx.rounds)
    {
        Apply(first);
    }

    void Update(sa::Engine& engine, double dt) override
    {
        sa::Backend& b = engine.GetBackend();
        mouse_.Update(b);
        if (mouse_.JustPressed())
            answerFocused_ = Answering() && !waiting_ && AnswerField().Contains(mouse_.Position());
        if (Answering() && !waiting_ && b.KeyPressed(sa::Key::Enter))
            answerFocused_ = true;
        ++frame_;
        UpdateSprites(b, static_cast<float>(dt));
        if (b.KeyPressed(sa::Key::Escape))
        {
            if (answerFocused_)
            {
                answerFocused_ = false;
                return;
            }
            engine.Quit();
            return;
        }
        if (sceneOpen_)
            sceneShown_ += kScenePointsPerFrame;
        if (waiting_)
        {
            PollReply(engine);
            return;
        }
        // Autoplay takes a turn 50 frames after the screen last changed, so captures see each step.
        const bool autoTurn = ctx_.autoplay && frame_ - frameOfLastChange_ == 50;
        if (UpdateScenes(autoTurn))
            return;
        const auto buttons = Buttons();
        if (Over())
            UpdateOver(engine, buttons, autoTurn);
        else if (Asking())
            UpdateQuestion(b, buttons, autoTurn);
        else
            UpdateSale(engine, buttons, autoTurn);
    }

    void Render(sa::Engine& engine) override
    {
        sa::Backend& b = engine.GetBackend();
        stage::DrawWorld(b);
        stage::DrawOnCloth(b, ctx_.drawing, ui::kInk);
        ctx_.girl.Draw(b);
        // Only the waiting question gets a bubble, over its buyer: one bubble cannot cover another, and a
        // buyer's reaction is not replaced by the question they ask next. Every line is read in the panel.
        const std::string asker = Answering() ? state_["question"].value("asker", std::string()) : std::string();
        std::map<std::string, std::string> question;
        if (const auto said = buyerSpeech_.find(asker); said != buyerSpeech_.end())
            question[asker] = said->second;
        CrowdView crowd{&state_["buyers"], &ctx_.art, &sprites_, &walkers_, mouse_.Position(), &question, asker,
                        HasScenes(), 5};
        DrawCrowd(b, crowd);
        ui::Title(b, "Selling: " + Noun());
        if (sceneOpen_)
            RenderScene(b);
        else
            RenderConversation(b);
        RenderSide(b);
        if (!error_.empty())
        {
            const auto lines = WrapLines(error_, kLogW - 28, 3);
            ui::Panel(b, {kLogX, kLogBottom + 10}, {kLogW, lines.size() * sa::LineHeight(1.5f) + 16});
            for (size_t i = 0; i < lines.size(); ++i)
                sa::DrawText(b, {kLogX + 14, kLogBottom + 18 + i * sa::LineHeight(1.5f)}, lines[i], 1.5f, ui::kBad);
        }
    }

private:
    void UpdateSprites(sa::Backend& b, float dt)
    {
        stage::Load(b);
        npcs::Load(b);
        ctx_.girl.Update(b, dt, !answerFocused_);
        ctx_.shelter.Update(b, dt);
        if (!ctx_.art.Loaded())
        {
            ctx_.art.Load(b, AssetDir() + "/characters");
            if (ctx_.autoplay)
                std::cout << "art: " << ctx_.art.LookCount() << " looks\n";
        }
        size_t index = 0;
        for (const auto& buyer : state_["buyers"])
        {
            const std::string name = buyer["name"];
            // The first buyers come as painted NPCs, one look each, a moment apart; the rest as pixel sprites.
            if (index < npcs::Count())
            {
                auto walker = walkers_.try_emplace(name, index, 0.7f * static_cast<float>(index)).first;
                walker->second.Update(dt, buyer["present"].get<bool>());
                ++index;
                continue;
            }
            ++index;
            if (const Look* look = ctx_.art.LookFor(name))
            {
                BuyerSprite& sprite = sprites_[name];
                sprite.SetMood(buyer.value("mood", "idle"), *look);
                sprite.Update(dt, *look);
            }
        }
    }

    // --- buyer scenes (docs/design-v2.md §5.4) ---------------------------------------------------

    // Scenes exist only while buyers are actually negotiating over a sellable item.
    bool HasScenes() const { return !Over() && state_.contains("noun"); }

    // Each buyer still at the table (their figure and tag) opens their scene; while one is shown, a Close
    // button in its corner shuts it (so does clicking that buyer again).
    std::vector<std::pair<std::string, ui::Button>> SceneButtons() const
    {
        std::vector<std::pair<std::string, ui::Button>> out;
        if (!HasScenes())
            return out;
        const auto spots = CrowdSpots(state_["buyers"]);
        for (size_t i = 0; i < spots.size(); ++i)
            if (state_["buyers"][i]["present"].get<bool>())
            {
                ui::Button hit = spots[i].hit;
                hit.enabled = !waiting_;
                out.push_back({spots[i].name, hit});
            }
        if (sceneOpen_)
            out.push_back({sceneBuyer_, CloseButton()});
        return out;
    }

    ui::Button CloseButton() const { return {{kLogX + kSceneW - 110, kLogTop + 10}, {96, 32}, "Close", !waiting_}; }

    // A click on a Scene button opens that buyer's scene (or closes it if it is the one shown).
    // Autoplay opens the first buyer's scene before its first answer and closes it on the next turn.
    // True when this frame was spent on scenes.
    bool UpdateScenes(bool autoTurn)
    {
        if (ctx_.autoplay && autoTurn && Asking() && HasScenes() && autoScene_ < 2)
        {
            if (autoScene_++ == 0)
                RequestScene(state_["buyers"][0]["name"]);
            else
            {
                sceneOpen_ = false;
                frameOfLastChange_ = frame_;
            }
            return true;
        }
        for (const auto& [name, button] : SceneButtons())
            if (mouse_.Clicked(button))
            {
                if (sceneOpen_ && sceneBuyer_ == name)
                    sceneOpen_ = false;
                else
                    RequestScene(name);
                return true;
            }
        return false;
    }

    void RequestScene(const std::string& name)
    {
        sceneBuyer_ = name;
        Send({{"type", "scene"}, {"buyer", name}, {"seed", sceneSeed_++}});
    }

    void ShowScene(const nlohmann::json& reply)
    {
        scene_ = reply;
        sceneOpen_ = true;
        sceneShown_ = 0;
        if (ctx_.autoplay)
        {
            std::cout << "scene for " << reply.value("buyer", std::string()) << ":";
            for (const auto& o : reply["objects"])
                std::cout << ' ' << o["label"].get<std::string>();
            std::cout << '\n';
        }
    }

    // The scene over the conversation and the stall: the drawing on the left, drawn a few points per frame
    // (item first, then each prop), and on the right what is in it and why.
    void RenderScene(sa::Backend& b) const
    {
        const float bottom = kSceneBottom;
        ui::Panel(b, {kLogX, kLogTop}, {kSceneW, bottom - kLogTop});
        sa::DrawText(b, {kLogX + 15, kLogTop + 14}, "How " + sceneBuyer_ + " pictures it at home", 2.0f, ui::kInk);
        ui::DrawButton(b, CloseButton(), mouse_.Position(), 1.6f);
        const float sceneW = scene_.value("width", 512.0f), sceneH = scene_.value("height", 288.0f);
        const float boxTop = kLogTop + 44, boxH = bottom - boxTop - 12;
        const float k = boxH / sceneH;
        const sa::Vec2 origin{kLogX + 15, boxTop};
        b.DrawRect(origin, {sceneW * k, sceneH * k}, ui::kWhite);
        int left = sceneShown_;
        for (const auto& o : scene_["objects"])
            for (const auto& stroke : o["strokes"])
            {
                const auto& xs = stroke[0];
                const auto& ys = stroke[1];
                for (size_t i = 0; i < xs.size() && left > 0; ++i, --left)
                {
                    const sa::Vec2 p{origin.x + xs[i].get<float>() * k, origin.y + ys[i].get<float>() * k};
                    if (i == 0)
                        b.DrawLine(p, p, 3.0f, ui::kInk);
                    else
                        b.DrawLine({origin.x + xs[i - 1].get<float>() * k, origin.y + ys[i - 1].get<float>() * k},
                                   p, 3.0f, ui::kInk);
                }
            }
        // What is in the picture and why, next to it.
        const float textX = origin.x + sceneW * k + 20;
        const float textW = kLogX + kSceneW - 15 - textX;
        // The buyer GPT says one line for the whole scene and the server puts it on every prop: then the
        // things are listed and the line is shown once, in full, under them.
        std::string shared;
        bool oneLine = true;
        for (const auto& o : scene_["objects"])
            if (o.value("role", "") != "item")
            {
                const std::string why = o.value("reason", "");
                if (shared.empty())
                    shared = why;
                else if (why != shared)
                    oneLine = false;
            }
        oneLine = oneLine && !shared.empty();
        float ty = boxTop + 4;
        for (const auto& o : scene_["objects"])
        {
            const std::string label = o["label"];
            const std::string why = o.value("role", "") == "item" ? std::string("for sale")
                                    : oneLine                     ? std::string()
                                                                  : o.value("reason", "");
            sa::DrawText(b, {textX, ty}, label, 2.0f, ui::kInk);
            ty += sa::LineHeight(2.0f);
            for (const auto& line : WrapLines(why, textW, 3))
            {
                sa::DrawText(b, {textX, ty}, line, 1.5f, ui::kMuted);
                ty += sa::LineHeight(1.5f);
            }
            ty += 8;
        }
        if (oneLine)
        {
            ty += 4;
            const float room = bottom - 12 - ty;
            const auto lines = WrapLines("\"" + shared + "\"", textW, static_cast<size_t>(room / sa::LineHeight(1.6f)), 1.6f);
            for (const auto& line : lines)
            {
                sa::DrawText(b, {textX, ty}, line, 1.6f, ui::kInk);
                ty += sa::LineHeight(1.6f);
            }
        }
    }

    // The left panel: every line of the last exchange in full (your answer, each buyer's reply, the next
    // question), so nothing is cut short in a bubble. The oldest lines give way when there is no room.
    void RenderConversation(sa::Backend& b) const
    {
        if (exchange_.empty())
            return;
        constexpr float kScale = 1.4f, kWidth = 230.0f, kGap = 6.0f, kRoom = 300.0f;
        const float lineH = sa::LineHeight(kScale);
        const float headH = sa::LineHeight(1.6f) + kGap;
        std::vector<std::pair<std::vector<std::string>, sa::Color>> blocks;
        float used = headH;
        for (size_t i = exchange_.size(); i-- > 0;)
        {
            const std::string& text = exchange_[i];
            const bool mine = text.rfind("You:", 0) == 0;
            const bool asking = Answering() && i + 1 == exchange_.size();
            const size_t fits = static_cast<size_t>((kRoom - used) / lineH);
            if (fits == 0)
                break;
            auto lines = WrapLines(text, kWidth, fits, kScale);
            used += lines.size() * lineH + kGap;
            blocks.insert(blocks.begin(), {std::move(lines), mine ? ui::kAccent : asking ? ui::kGood : ui::kInk});
        }
        // The panel grows with the text, leaving its frame and flowers clear.
        const sa::Vec2 size = ui::PanelFit(b, {kWidth, used});
        const sa::Vec2 inset = ui::PanelInset(b, size);
        ui::Panel(b, {kLogX, kLogTop}, size);
        float y = kLogTop + inset.y;
        const float x = kLogX + inset.x;
        sa::DrawText(b, {x, y}, "This round", 1.6f, ui::kMuted);
        y += headH;
        for (const auto& [lines, color] : blocks)
        {
            for (const auto& line : lines)
            {
                sa::DrawText(b, {x, y}, line, kScale, color);
                y += lineH;
            }
            y += kGap;
        }
    }

    // The right panel: what to do now (heading), the answer field while a question waits, and the buttons.
    std::vector<std::string> SideHeading() const
    {
        if (Answering())
            return {"Your answer"};
        std::vector<std::string> lines = WrapLines(!result_.empty()   ? result_
                                                   : Over()           ? "Nobody wants that. Try drawing something else."
                                                   : Offers().empty() ? "Everyone walked away."
                                                                      : "Who do you sell to?",
                                                   kSideW - kSideHeadLeft - kSidePad, 4, 1.8f);
        if (!result_.empty())
            lines.insert(lines.begin(), "Sold!");
        return lines;
    }

    float SideContentTop() const
    {
        const float heading = kSideTop + kSideHeadTop + SideHeading().size() * sa::LineHeight(1.8f);
        return Answering() ? heading + kFieldH + 10 : heading + 4;
    }

    ui::Button AnswerField() const
    {
        return {{kSideX + kSidePad, kSideTop + kSideHeadTop + SideHeading().size() * sa::LineHeight(1.8f)},
                {kSideW - 2 * kSidePad, kFieldH}, ""};
    }

    void RenderSide(sa::Backend& b) const
    {
        const auto buttons = Buttons();
        const float bottom = buttons.empty() ? SideContentTop() : buttons.back().pos.y + kButtonH;
        ui::Panel(b, {kSideX, kSideTop}, {kSideW, bottom + 40 - kSideTop});   // 40: the bottom frame and flowers
        float y = kSideTop + kSideHeadTop;
        for (const std::string& line : SideHeading())
        {
            sa::DrawText(b, {kSideX + kSideHeadLeft, y}, line, 1.8f, line == "Sold!" ? ui::kGood : ui::kMuted);
            y += sa::LineHeight(1.8f);
        }
        if (Answering())
            RenderAnswerField(b, {kSideX + kSidePad, y});
        for (const auto& button : buttons)
            ui::DrawButton(b, button, mouse_.Position(), 1.8f);
    }

    // Where the player types an answer. Empty, it says what the last typed answer was taken as (or that
    // nobody could tell), so a wrong guess is visible right away; otherwise how to use it.
    void RenderAnswerField(sa::Backend& b, sa::Vec2 pos) const
    {
        const float w = kSideW - 2 * kSidePad;
        // A plain white field: the flowered skin would shrink around 36px and cover the typed text.
        if (!stage::DrawRounded(b, pos, {w, kFieldH}, ui::kWhite, ui::kBorder))
            b.DrawRect(pos, {w, kFieldH}, ui::kWhite);
        const sa::Vec2 text{pos.x + kFieldPad, pos.y + (kFieldH - sa::kGlyphSize * kFieldScale) / 2};
        const float cursorW = sa::TextWidth("_", kFieldScale);
        const bool cursor = answerFocused_ && !waiting_ && (frame_ / 30) % 2 == 0;
        float cursorX = text.x;
        if (input_.empty())
            RenderFieldHint(b, {text.x + cursorW + 6, pos.y}, w - 2 * kFieldPad - cursorW - 6);
        else
        {
            // The end of what was typed, if it is wider than the field.
            std::string shown = input_;
            while (!shown.empty() && sa::TextWidth(shown, kFieldScale) + cursorW > w - 2 * kFieldPad)
                shown.erase(0, 1);
            sa::DrawText(b, text, shown, kFieldScale, ui::kInk);
            cursorX += sa::TextWidth(shown, kFieldScale);
        }
        if (cursor)
            sa::DrawText(b, {cursorX, text.y}, "_", kFieldScale, ui::kAccent);
    }

    void RenderFieldHint(sa::Backend& b, sa::Vec2 at, float width) const
    {
        const std::string hint = !fieldNote_.empty() ? fieldNote_ :
            answerFocused_ ? "Type, then Enter. Esc to move." : "Click here / Enter to type";
        const auto lines = WrapLines(hint, width, 1, 1.4f);
        if (!lines.empty())
            sa::DrawText(b, {at.x, at.y + (kFieldH - sa::kGlyphSize * 1.4f) / 2}, lines[0], 1.4f,
                         fieldNoteBad_ ? ui::kBad : ui::kMuted);
    }

    // A buyer's question waits and the round has not ended.
    bool Answering() const { return Asking() && !Over(); }

    // The round is over: sold, or the item was refused. One button: draw something new.
    bool Over() const { return !result_.empty() || state_["label"] == "refuse"; }
    // A buyer's question is waiting for the player's answer.
    bool Asking() const { return !state_["finished"].get<bool>(); }

    void PollReply(sa::Engine& engine)
    {
        const auto reply = ctx_.client.Poll();
        if (!reply)
            return;
        waiting_ = false;
        frameOfLastChange_ = frame_;
        const std::string type = reply->value("type", "");
        if (type == "scene_result")
            ShowScene(*reply);
        else if (type == "round_state")
            Apply(*reply);
        else if (type == "round_result")
            Sold(*reply);
        else
            error_ = ReplyError(ctx_, engine, *reply);
    }

    void UpdateOver(sa::Engine& engine, const std::vector<ui::Button>& buttons, bool autoTurn)
    {
        if (!mouse_.Clicked(buttons[0]) && !autoTurn)
            return;
        if (ctx_.autoplay)
            engine.Quit();
        else
            engine.SetScene(MakeCanvasScene(ctx_));
    }

    // A typed answer (Enter sends it) or a clicked option.
    void UpdateQuestion(sa::Backend& b, const std::vector<ui::Button>& buttons, bool autoTurn)
    {
        if (ReadTyping(b))
        {
            Send({{"type", "round_answer"}, {"text", input_}});
            return;
        }
        const auto& options = state_["question"]["options"];
        if (autoTurn)
        {
            AutoAnswer(options);
            return;
        }
        for (size_t i = 0; i < options.size(); ++i)
            if (mouse_.Clicked(buttons[i]))
            {
                Send({{"type", "round_answer"}, {"option", options[i]["id"]}});
                return;
            }
    }

    // Adds this frame's typing to the field; true when Enter was pressed on something to send.
    bool ReadTyping(sa::Backend& b)
    {
        if (!answerFocused_)
            return false;
        for (char c : b.TypedText())
            if (input_.size() < kMaxTyped)
                input_ += c;
        if (b.KeyPressed(sa::Key::Backspace) && !input_.empty())
            input_.pop_back();
        return b.KeyPressed(sa::Key::Enter) && input_.find_first_not_of(' ') != std::string::npos;
    }

    // Autoplay picks a random option and alternates clicking it with typing its words in lower case,
    // so a test round goes through both paths.
    void AutoAnswer(const nlohmann::json& options)
    {
        const auto& option = options[std::uniform_int_distribution<size_t>(0, options.size() - 1)(rng_)];
        if (autoTurns_++ % 2 == 0)
        {
            Send({{"type", "round_answer"}, {"option", option["id"]}});
            return;
        }
        std::string typed = option["answer"];
        std::transform(typed.begin(), typed.end(), typed.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        Send({{"type", "round_answer"}, {"text", typed}});
    }

    // Questions are done: one button per buyer still here (best offer first), then "Keep it".
    void UpdateSale(sa::Engine& engine, const std::vector<ui::Button>& buttons, bool autoTurn)
    {
        const auto offers = Offers();
        for (size_t i = 0; i < offers.size(); ++i)
            if (mouse_.Clicked(buttons[i]) || (autoTurn && i == 0))
            {
                Send({{"type", "round_sell"}, {"buyer", offers[i].first}});
                return;
            }
        if (autoTurn && offers.empty())
        {
            std::cout << "autoplay: every buyer left\n";
            engine.Quit();
        }
        else if (mouse_.Clicked(buttons.back()))
            engine.SetScene(MakeCanvasScene(ctx_));
    }

    void Sold(const nlohmann::json& reply)
    {
        if (reply.contains("state"))
            state_ = reply["state"];        // the buyer's mood is now "won"
        result_ = reply["buyer"].get<std::string>() + " buys your " + Noun() + " for " +
                  Money(reply["price"]) + " (a typical one goes for " + Money(reply["typical"]) + ").";
        log_.push_back(result_);
        ctx_.sold = true;
        if (ctx_.autoplay)
            std::cout << result_ << '\n';
    }

    void Send(nlohmann::json msg)
    {
        if (msg["type"] != "scene")
            sceneOpen_ = false;      // answering or selling brings the conversation back
        if (msg["type"] == "round_answer")
            ctx_.girl.Feel(Girl::Face::Sheepish, 30.0f);   // until the buyers answer
        msg["session"] = ctx_.session;
        ctx_.client.Send(std::move(msg));
        waiting_ = true;
        error_.clear();
    }

    void Apply(const nlohmann::json& reply)
    {
        answerFocused_ = false;
        Feelings(state_, reply["state"]);
        state_ = reply["state"];
        fieldNote_.clear();
        fieldNoteBad_ = false;
        if (reply.contains("heard"))            // the reply to a typed answer
        {
            input_.clear();
            if (reply["heard"].is_null())
            {
                fieldNote_ = "Not sure which one you meant. Try other words, or click one:";
                fieldNoteBad_ = true;
            }
            else
                fieldNote_ = "Taken as: " + reply["heard"]["answer"].get<std::string>();
            if (ctx_.autoplay)
                std::cout << "heard: " << (reply["heard"].is_null() ? "(asked again)" : reply["heard"].dump()) << '\n';
        }
        if (!reply["new_lines"].empty())
            exchange_.clear();
        for (const auto& line : reply["new_lines"])
        {
            exchange_.push_back(line);
            const std::string text = line;
            log_.push_back(text);
            const size_t colon = text.find(": ");
            if (colon != std::string::npos && text.rfind("You:", 0) != 0)
                buyerSpeech_[text.substr(0, colon)] = text.substr(colon + 2);
            if (ctx_.autoplay)
                std::cout << text << '\n';      // the transcript, for test logs
        }
    }

    // The girl's face after the buyers' reactions: anxious when someone walked away, cross when someone
    // lowered their offer, otherwise back to her usual self.
    void Feelings(const nlohmann::json& before, const nlohmann::json& after)
    {
        ctx_.girl.Calm();
        if (!before.is_object() || !before.contains("buyers"))
            return;
        bool left = false, lowered = false;
        for (const auto& buyer : after["buyers"])
        {
            for (const auto& old : before["buyers"])
                if (old["name"] == buyer["name"] && old["present"].get<bool>() && !buyer["present"].get<bool>())
                    left = true;
            const nlohmann::json moved = buyer.value("change", nlohmann::json());
            if (buyer["present"].get<bool>() && moved.is_object() && moved.value("amount", 0.0) <= -0.5)
                lowered = true;
        }
        if (left)
            ctx_.girl.Feel(Girl::Face::Anxious);
        else if (lowered)
            ctx_.girl.Feel(Girl::Face::Angry);
    }

    std::string Noun() const { return state_.value("noun", state_.value("item", std::string("item"))); }

    std::vector<std::pair<std::string, double>> Offers() const
    {
        std::vector<std::pair<std::string, double>> out;
        for (const auto& buyer : state_["buyers"])
            if (buyer["present"].get<bool>() && !buyer["offer"].is_null())
                out.emplace_back(buyer["name"], buyer["offer"]);
        std::sort(out.begin(), out.end(), [](auto& a, auto& b) { return a.second > b.second; });
        return out;
    }

    std::vector<ui::Button> Buttons() const
    {
        std::vector<std::string> labels;
        if (Over())
            labels.push_back("Draw something new");
        else if (Asking())
            for (const auto& o : state_["question"]["options"])
                labels.push_back(o["answer"].get<std::string>());
        else
        {
            for (const auto& [name, offer] : Offers())
                labels.push_back("Sell to " + name + " for " + Money(offer));
            labels.push_back("Keep it");
        }
        // One column under the heading (and the answer field while a question waits).
        const float w = kSideW - 2 * kSidePad;
        const float top = SideContentTop();
        std::vector<ui::Button> out;
        for (size_t i = 0; i < labels.size(); ++i)
            out.push_back({{kSideX + kSidePad, top + i * (kButtonH + kButtonGap)}, {w, kButtonH}, labels[i], !waiting_});
        return out;
    }

    GameContext& ctx_;
    ui::Mouse mouse_;
    nlohmann::json state_;
    std::vector<std::string> log_;
    std::string result_;
    std::string error_;
    std::string input_;           // what the player is typing
    bool answerFocused_ = false;
    std::string fieldNote_;       // shown in the empty field: how the last typed answer was taken
    bool fieldNoteBad_ = false;
    int autoTurns_ = 0;
    bool waiting_ = false;
    int frame_ = 0;
    int frameOfLastChange_ = 0;
    std::mt19937 rng_;
    std::map<std::string, BuyerSprite> sprites_;
    std::map<std::string, Walker> walkers_;
    std::map<std::string, std::string> buyerSpeech_; // latest visible line for every buyer
    std::vector<std::string> exchange_;               // the lines of the last exchange, shown in full
    nlohmann::json scene_;        // the last scene_result
    std::string sceneBuyer_;      // whose scene is shown (or was asked for)
    bool sceneOpen_ = false;
    int sceneShown_ = 0;          // points drawn so far
    int sceneSeed_ = 1;           // a different drawing each time a scene is opened
    int autoScene_ = 0;           // autoplay: 0 = not yet, 1 = shown, 2 = closed
};

} // namespace

std::unique_ptr<sa::Scene> MakeRoundScene(GameContext& ctx, nlohmann::json first)
{
    return std::make_unique<RoundScene>(ctx, std::move(first));
}
