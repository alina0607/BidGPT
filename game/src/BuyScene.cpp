#include "Crowd.h"
#include "Scenes.h"
#include "Stage.h"
#include "Ui.h"

#include "sa/Text.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace
{

// Buying a cat (docs/protocol.md, "Buying a cat"): the model is the seller and the other buyers, the player is the
// last buyer in the row. The same panels as the selling round: the sale's lines on the left, the player's turn on
// the right (what to say, the offer, and the buttons).
constexpr float kLogX = 20.0f;
constexpr float kLogTop = 78.0f;
constexpr float kLogW = 300.0f;
constexpr float kLogBottom = 425.0f;    // above the leftmost buyer's bubble
constexpr float kLogTextW = 200.0f;     // inside the panel's frame and flowers
constexpr float kLineScale = 1.4f;
constexpr float kSideX = 960.0f;
constexpr float kSideW = 300.0f;
constexpr float kSideTop = 78.0f;
constexpr float kSidePad = 14.0f;
constexpr float kSideHeadTop = 40.0f;   // below the panel's top frame and flowers (as in RoundScene)
constexpr float kSideHeadLeft = 62.0f;
constexpr float kFieldH = 36.0f;
constexpr float kFieldPad = 12.0f;
constexpr float kFieldScale = 1.8f;
constexpr float kButtonH = 40.0f;
constexpr float kButtonGap = 8.0f;
constexpr size_t kMaxTyped = 120;       // the server keeps at most this many characters too
constexpr int kModelBuyers = 3;
constexpr int kAutoDelay = 50;          // autoplay's frames between steps
constexpr size_t kBubbleLines = 4;      // the most lines a buyer's bubble shows
constexpr float kStepGap = 6.0f;        // between the offer's step buttons
constexpr float kStepH = 38.0f;
constexpr float kOfferLineH = 30.0f;    // "Your offer: $N" above the step buttons
const char* const kPlayer = "You";      // the player's name at the stall: their lines read "You: ..."

// A button and what it does.
struct Action
{
    std::string id;
    ui::Button button;
};

class BuyScene final : public sa::Scene
{
public:
    explicit BuyScene(GameContext& ctx) : ctx_(ctx) {}

    // One frame: the market moves, then the server's reply or the player's turn (autoplay takes it 50 frames
    // after the screen last changed, so captures see each step).
    void Update(sa::Engine& engine, double dt) override
    {
        sa::Backend& b = engine.GetBackend();
        mouse_.Update(b);
        ++frame_;
        stage::Load(b);
        ctx_.girl.Update(b, static_cast<float>(dt), !focused_);
        ctx_.shelter.Update(b, static_cast<float>(dt));
        if (state_.is_object())
            UpdateCrowdSprites(b, ctx_.art, state_["buyers"], sprites_, walkers_, static_cast<float>(dt),
                               ctx_.autoplay);
        if (b.KeyPressed(sa::Key::Escape))
        {
            if (focused_)
                focused_ = false;
            else
                engine.Quit();
            return;
        }
        if (!started_)
        {
            Start();
            return;
        }
        if (waiting_)
        {
            PollReply(engine);
            return;
        }
        if (ctx_.autoplay && frame_ - frameOfLastChange_ == kAutoDelay)
            AutoTurn(engine);
        else
            UpdateTurn(engine, b);
    }

    // The scene drawn over the market: the buyers, the sale's lines and the player's turn.
    void Render(sa::Engine& engine) override
    {
        sa::Backend& b = engine.GetBackend();
        stage::DrawWorld(b);
        ctx_.girl.Draw(b);
        if (state_.is_object())
        {
            CrowdView crowd{&state_["buyers"], &ctx_.art, &sprites_, &walkers_, mouse_.Position(), &speech_,
                            Asker(), false, kBubbleLines};
            DrawCrowd(b, crowd);
        }
        ui::Title(b, state_.is_object() ? "Buying a cat: asking " + Money(state_.value("asking", 0), false)
                                        : "Buying a cat");
        RenderLog(b);
        RenderSide(b);
        RenderError(b);
    }

private:
    // The player's input: focusing the field, typing (Enter sends), paging the lines, and the buttons.
    void UpdateTurn(sa::Engine& engine, sa::Backend& b)
    {
        if (mouse_.JustPressed())
            focused_ = Typing() && Field().Contains(mouse_.Position());
        else if (Typing() && !focused_ && b.KeyPressed(sa::Key::Enter))
        {
            focused_ = true;
            return;
        }
        if (ReadTyping(b) && Step() != "reply")
        {
            Act(engine, Step() == "ask" ? "ask" : "send");
            return;
        }
        if (mouse_.Clicked(EarlierButton()))
            top_ -= std::min(top_, visible_ - 1);
        else if (mouse_.Clicked(LaterButton()))
            top_ = std::min(top_ + visible_ - 1, wrapped_.size() - std::min(visible_, wrapped_.size()));
        for (const Action& a : Actions())
            if (mouse_.Clicked(a.button))
            {
                Act(engine, a.id);
                return;
            }
    }

    // A server error, under the sale's lines.
    void RenderError(sa::Backend& b) const
    {
        if (error_.empty())
            return;
        const auto lines = WrapLines(error_, kLogW - 28, 3);
        const float top = kLogBottom + 50, lineH = sa::LineHeight(1.5f);
        ui::Panel(b, {kLogX, top}, {kLogW, lines.size() * lineH + 16});
        for (size_t i = 0; i < lines.size(); ++i)
            sa::DrawText(b, {kLogX + 14, top + 8 + i * lineH}, lines[i], 1.5f, ui::kBad);
    }

    // --- talking to the server ------------------------------------------------------------------

    // The client takes one request at a time: a live guess from the drawing screen may still be in flight.
    void Start()
    {
        if (ctx_.client.Busy())
        {
            ctx_.client.Poll();
            return;
        }
        started_ = true;
        Send({{"type", "round_start"}, {"session", ctx_.NewSession()}, {"item", "cat"}, {"buyers", kModelBuyers},
              {"role", "buyer"}, {"name", kPlayer}});
    }

    void Send(nlohmann::json msg)
    {
        if (msg["type"] != "round_start")
            msg["session"] = ctx_.session;
        ctx_.client.Send(std::move(msg));
        waiting_ = true;
        focused_ = false;
        error_.clear();
    }

    // The player's step: what they typed, and the offer for an opening or a reaction (0 walks away).
    void SendStep(const std::string& step, const std::string& text, int offer = -1, bool accept = false)
    {
        nlohmann::json msg{{"type", "round_buy"}, {"step", step}, {"text", text}};
        if (offer >= 0)
            msg["offer"] = offer;
        if (step == "reply")
            msg["accept"] = accept;
        Send(std::move(msg));
    }

    // The server's reply, once in: the sale's new state, or an error.
    void PollReply(sa::Engine& engine)
    {
        const auto reply = ctx_.client.Poll();
        if (!reply)
            return;
        waiting_ = false;
        frameOfLastChange_ = frame_;
        if (reply->value("type", "") == "round_state")
            Apply(*reply);
        else
            error_ = ReplyError(ctx_, engine, *reply);
    }

    // A round_state: its new lines, the step it waits for (and that step's starting offer), and the result.
    void Apply(const nlohmann::json& reply)
    {
        state_ = reply["state"];
        input_.clear();
        note_.clear();
        noteBad_ = false;
        const size_t firstNew = wrapped_.size();
        for (const auto& line : reply["new_lines"])
            AddLine(line.get<std::string>());
        // Show the new lines from the first one, or the last page when they all fit with some earlier ones.
        top_ = std::min(firstNew, wrapped_.size() > visible_ ? wrapped_.size() - visible_ : size_t{0});
        if (Step() == "open")
            offer_ = std::max(5, state_.value("asking", 0));
        else if (Step() == "react")
            offer_ = MyOffer();
        if (Finished())
        {
            ctx_.sold = true;           // the sale ran to the seller's decisions (autoplay's pass)
            if (ctx_.autoplay)
                std::cout << Result() << '\n';
        }
    }

    // A line of the sale: shown in the left panel, and a buyer's latest line goes into their bubble.
    void AddLine(const std::string& text)
    {
        if (ctx_.autoplay)
            std::cout << text << '\n';
        const bool mine = text.rfind(std::string(kPlayer) + ":", 0) == 0;
        const bool seller = text.rfind("Seller:", 0) == 0;
        const sa::Color color = mine ? ui::kAccent : seller ? ui::kGood : ui::kInk;
        for (auto& line : WrapLines(text, kLogTextW, 12, kLineScale))
            wrapped_.emplace_back(std::move(line), color);
        wrapped_.emplace_back("", ui::kInk);     // a gap between lines
        const size_t colon = text.find(": ");
        if (colon != std::string::npos && !mine && !seller)
            speech_[text.substr(0, colon)] = text.substr(colon + 2);
    }

    // --- the player's turn ----------------------------------------------------------------------

    std::string Step() const
    {
        if (!state_.is_object() || state_["waiting"].is_null())
            return "";
        return state_["waiting"].get<std::string>();
    }
    bool Finished() const { return state_.is_object() && state_.value("finished", false); }
    bool Typing() const { return !waiting_ && !Step().empty(); }
    std::string Asker() const
    {
        return state_.is_object() && state_["question"].is_object() ? state_["question"].value("asker", "") : "";
    }

    int MyOffer() const
    {
        for (const auto& buyer : state_["buyers"])
            if (buyer["name"] == kPlayer && !buyer["offer"].is_null())
                return static_cast<int>(buyer["offer"].get<double>());
        return 0;
    }
    int CounterPrice() const
    {
        const auto& d = state_["decisions"];
        return d.contains(kPlayer) ? d[kPlayer].value("price", 0) : 0;
    }

    // A button (or Enter): change the offer, send the player's step, or go back to the stall.
    void Act(sa::Engine& engine, const std::string& id)
    {
        const std::string step = Step();
        const bool typed = input_.find_first_not_of(' ') != std::string::npos;
        if (id == "back")
            engine.SetScene(MakeCanvasScene(ctx_));
        else if (id.rfind("offer", 0) == 0)
            offer_ = std::max(1, offer_ + std::stoi(id.substr(5)));
        else if (id == "walk" && step == "react")
            SendStep("react", typed ? input_ : "I'll pass on this one.", 0);
        else if (id == "accept" || id == "walk")
            SendStep("reply", typed ? input_ : id == "accept" ? "Deal." : "No, thank you.", -1, id == "accept");
        else if (!typed)
        {
            note_ = "Type what you say first.";
            noteBad_ = true;
        }
        else if (step == "ask")
            SendStep("ask", input_);
        else
            SendStep(step, input_, offer_);
    }

    // Autoplay buys at the asking price less $10, asks one question, adds $5 after the answer and takes a counter.
    void AutoTurn(sa::Engine& engine)
    {
        const std::string step = Step();
        if (Finished() || (!waiting_ && step.empty()))
            engine.Quit();
        else if (step == "open")
            SendStep("open", "Hello! I'd love a calm cat for my small flat.", std::max(5, state_.value("asking", 0) - 10));
        else if (step == "ask")
            SendStep("ask", "Has she seen a vet lately, and is she spayed?");
        else if (step == "react")
            SendStep("react", "Thanks, that helps.", MyOffer() + 5);
        else if (step == "reply")
            SendStep("reply", "Deal.", -1, true);
    }

    // Adds this frame's typing to the field; true when Enter was pressed on something to send.
    bool ReadTyping(sa::Backend& b)
    {
        if (!focused_ || !Typing())
            return false;
        for (char c : b.TypedText())
            if (input_.size() < kMaxTyped)
                input_ += c;
        if (b.KeyPressed(sa::Key::Backspace) && !input_.empty())
            input_.pop_back();
        return b.KeyPressed(sa::Key::Enter) && input_.find_first_not_of(' ') != std::string::npos;
    }

    // --- the right panel ------------------------------------------------------------------------

    // Who bought the cat and for how much.
    std::string Result() const
    {
        const auto& sold = state_["sold"];
        const std::string asking = " (asking " + Money(sold.value("asking", 0), false) + ")";
        if (sold["buyer"].is_null())
            return "No deal: the cat stays with the seller.";
        const std::string price = Money(sold["price"].get<double>(), false);
        if (sold["buyer"] == kPlayer)
            return "You bought the cat for " + price + asking + "!";
        return sold["buyer"].get<std::string>() + " bought the cat for " + price + asking + ".";
    }

    // The right panel's heading: what the player does now, or what is happening.
    std::vector<std::string> Heading() const
    {
        const std::string step = Step();
        const std::string text = !started_ || (waiting_ && !state_.is_object()) ? "The seller is bringing a cat..."
                                 : waiting_                                    ? "The others are talking..."
                                 : Finished()                                  ? Result()
                                 : step == "open"  ? "Say hello and make your first offer"
                                 : step == "ask"   ? "Your turn: ask the seller one question"
                                 : step == "react" ? "What you say, and your offer now"
                                 : step == "reply" ? "The seller counters at " + Money(CounterPrice(), false) + "."
                                                   : "";
        return WrapLines(text, kSideW - kSideHeadLeft - kSidePad, 4, 1.8f);
    }

    float FieldTop() const { return kSideTop + kSideHeadTop + Heading().size() * sa::LineHeight(1.8f); }
    ui::Button Field() const { return {{kSideX + kSidePad, FieldTop()}, {kSideW - 2 * kSidePad, kFieldH}, ""}; }
    bool HasOffer() const { return Step() == "open" || Step() == "react"; }
    float OfferTop() const { return FieldTop() + kFieldH + 10; }

    // The buttons of the player's turn (none while the others talk): the offer's steps, then what to do.
    std::vector<Action> Actions() const
    {
        std::vector<Action> out;
        if (waiting_)
            return out;
        if (HasOffer())
            out = OfferSteps();
        const float x = kSideX + kSidePad, w = kSideW - 2 * kSidePad;
        float top = Finished() || Step().empty() ? FieldTop() + 6
                    : HasOffer()                 ? OfferTop() + kOfferLineH + kStepH + 10
                                                 : FieldTop() + kFieldH + 10;
        for (const auto& [id, label] : TurnLabels())
        {
            out.push_back({id, {{x, top}, {w, kButtonH}, label}});
            top += kButtonH + kButtonGap;
        }
        return out;
    }

    // Four steps across the field's width, under "Your offer".
    std::vector<Action> OfferSteps() const
    {
        const char* const steps[] = {"-5", "-1", "+1", "+5"};
        const float w = kSideW - 2 * kSidePad, stepW = (w - 3 * kStepGap) / 4;
        std::vector<Action> out;
        for (int i = 0; i < 4; ++i)
            out.push_back({std::string("offer") + steps[i],
                           {{kSideX + kSidePad + i * (stepW + kStepGap), OfferTop() + kOfferLineH}, {stepW, kStepH},
                            steps[i], true, ui::ButtonStyle::Cream}});
        return out;
    }

    // What the player can do now, as (action, label).
    std::vector<std::pair<std::string, std::string>> TurnLabels() const
    {
        const std::string step = Step();
        std::vector<std::pair<std::string, std::string>> labels;
        if (Finished())
            labels = {{"back", "Back to the stall"}};
        else if (step == "open" || step == "react")
            labels = {{"send", "Say it"}};
        else if (step == "ask")
            labels = {{"ask", "Ask"}};
        else if (step == "reply")
            labels = {{"accept", "Accept " + Money(CounterPrice(), false)}};
        if (step == "react" || step == "reply")
            labels.push_back({"walk", "Walk away"});
        return labels;
    }

    // The right panel: heading, field, offer and buttons.
    void RenderSide(sa::Backend& b) const
    {
        const auto actions = Actions();
        const float bottom = actions.empty() ? FieldTop() + (Typing() ? kFieldH : 0) : actions.back().button.pos.y + kButtonH;
        ui::Panel(b, {kSideX, kSideTop}, {kSideW, bottom + 40 - kSideTop});
        float y = kSideTop + kSideHeadTop;
        for (const std::string& line : Heading())
        {
            sa::DrawText(b, {kSideX + kSideHeadLeft, y}, line, 1.8f, Finished() ? ui::kGood : ui::kMuted);
            y += sa::LineHeight(1.8f);
        }
        if (Typing())
            RenderField(b);
        if (HasOffer() && !waiting_)
            sa::DrawText(b, {kSideX + kSideHeadLeft, OfferTop() + 4}, "Your offer: " + Money(offer_, false), 1.8f,
                         ui::kInk);
        for (const Action& a : actions)
            ui::DrawButton(b, a.button, mouse_.Position(), 1.8f);
    }

    // Where the player types, with a blinking cursor while focused.
    void RenderField(sa::Backend& b) const
    {
        const ui::Button f = Field();
        if (!stage::DrawRounded(b, f.pos, f.size, ui::kWhite, ui::kBorder))
            b.DrawRect(f.pos, f.size, ui::kWhite);
        const sa::Vec2 text{f.pos.x + kFieldPad, f.pos.y + (kFieldH - sa::kGlyphSize * kFieldScale) / 2};
        const float room = f.size.x - 2 * kFieldPad - sa::TextWidth("_", kFieldScale);
        float cursorX = text.x;
        if (input_.empty())
            RenderHint(b, {text.x + 14, f.pos.y}, room);
        else
        {
            std::string shown = input_;     // the end of what was typed, if it is wider than the field
            while (!shown.empty() && sa::TextWidth(shown, kFieldScale) > room)
                shown.erase(0, 1);
            sa::DrawText(b, text, shown, kFieldScale, ui::kInk);
            cursorX += sa::TextWidth(shown, kFieldScale);
        }
        if (focused_ && (frame_ / 30) % 2 == 0)
            sa::DrawText(b, {cursorX, text.y}, "_", kFieldScale, ui::kAccent);
    }

    // In the empty field: the last note (a refused send), or how to type.
    void RenderHint(sa::Backend& b, sa::Vec2 at, float width) const
    {
        const std::string hint = !note_.empty() ? note_ : focused_ ? "Type, then Enter." : "Click here to type";
        const auto lines = WrapLines(hint, width, 1, 1.4f);
        if (!lines.empty())
            sa::DrawText(b, {at.x, at.y + (kFieldH - sa::kGlyphSize * 1.4f) / 2}, lines[0], 1.4f,
                         noteBad_ ? ui::kBad : ui::kMuted);
    }

    // --- the left panel: every line of the sale, a page at a time ---------------------------------

    ui::Button EarlierButton() const { return {{kLogX + 14, kLogBottom - 46}, {120, 34}, "Earlier", top_ > 0}; }
    ui::Button LaterButton() const
    {
        return {{kLogX + kLogW - 134, kLogBottom - 46}, {120, 34}, "Later", top_ + visible_ < wrapped_.size()};
    }

    // The left panel: a page of the sale's lines (measuring how many fit), with Earlier / Later.
    void RenderLog(sa::Backend& b)
    {
        if (wrapped_.empty())
            return;
        const sa::Vec2 size{kLogW, kLogBottom - kLogTop};
        const sa::Vec2 inset = ui::PanelInset(b, size);
        ui::Panel(b, {kLogX, kLogTop}, size);
        const float lineH = sa::LineHeight(kLineScale);
        const float first = kLogTop + inset.y;
        visible_ = static_cast<size_t>((kLogBottom - 52 - first) / lineH);
        for (size_t i = top_; i < wrapped_.size() && i < top_ + visible_; ++i)
            sa::DrawText(b, {kLogX + inset.x, first + (i - top_) * lineH}, wrapped_[i].first, kLineScale,
                         wrapped_[i].second);
        ui::DrawButton(b, EarlierButton(), mouse_.Position(), 1.5f);
        ui::DrawButton(b, LaterButton(), mouse_.Position(), 1.5f);
    }

    GameContext& ctx_;
    ui::Mouse mouse_;
    nlohmann::json state_;
    bool started_ = false;
    bool waiting_ = false;
    bool focused_ = false;          // the player is typing in the field
    std::string input_;
    std::string note_;              // shown in the empty field
    bool noteBad_ = false;
    std::string error_;
    int offer_ = 0;
    int frame_ = 0;
    int frameOfLastChange_ = 0;
    std::vector<std::pair<std::string, sa::Color>> wrapped_;   // the sale's lines, wrapped for the left panel
    size_t top_ = 0;                // the first wrapped line shown
    size_t visible_ = 12;           // how many fit (measured when drawn)
    std::map<std::string, std::string> speech_;                // each buyer's latest line, for their bubble
    std::map<std::string, BuyerSprite> sprites_;
    std::map<std::string, Walker> walkers_;
};

} // namespace

std::unique_ptr<sa::Scene> MakeBuyScene(GameContext& ctx) { return std::make_unique<BuyScene>(ctx); }
