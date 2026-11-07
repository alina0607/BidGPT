#include "Crowd.h"
#include "Scenes.h"
#include "Stage.h"
#include "Ui.h"

#include "sa/Text.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace
{

// Comparing buyer models, or one model's ways of picking words, on one cat sale (docs/protocol.md "compare";
// ml/sellanything_ml/buyers/paired.py and variants.py): the same buyers hear the same question and the player's one
// answer, and every buyer's line from each model or way is shown next to the others'. While testing the data this covers the market (owner, 2026-10-28: the stall and the
// characters may be hidden); every sale is saved on the server under ml/logs/compare/.
constexpr float kPad = 16.0f;
constexpr float kLookTop = 46.0f;          // the turn in view: its question and the answer
constexpr float kNowTop = 88.0f;           // the question waiting for an answer
constexpr float kFieldTop = 132.0f;
constexpr float kFieldH = 36.0f;
constexpr float kFieldW = 1000.0f;
constexpr float kFieldPad = 12.0f;
constexpr float kFieldScale = 1.8f;
constexpr float kGridTop = 192.0f;
constexpr float kGridBottom = 700.0f;
constexpr float kHeadH = 26.0f;
constexpr float kInfoW = 300.0f;            // the buyer's column with two columns of lines; they share the rest
constexpr float kInfoWide = 200.0f;         // the buyer's column with three or more
constexpr float kTextScale = 1.6f;
constexpr float kInfoScale = 1.45f;        // the buyer's persona and what they care about
constexpr size_t kMaxTyped = 120;           // the server keeps at most this many characters too
constexpr int kAutoDelay = 50;              // autoplay's frames between steps
const char* const kAutoAnswers[] = {"she is calm and healthy", "no", "3 years old", "she is playful"};

struct Action
{
    std::string id;
    ui::Button button;
};

class CompareScene final : public sa::Scene
{
public:
    CompareScene(GameContext& ctx, std::string models) : ctx_(ctx)
    {
        std::stringstream list(models);
        for (std::string m; std::getline(list, m, ',');)
            if (!m.empty())
                models_.push_back(m);
    }

    // One frame: start the sale once the client is free, then the server's reply or the player's turn.
    void Update(sa::Engine& engine, double) override
    {
        sa::Backend& b = engine.GetBackend();
        mouse_.Update(b);
        ++frame_;
        stage::Load(b);             // the market's Nunito font (the 8x8 bitmap font is hard to read)
        if (b.KeyPressed(sa::Key::Escape))
        {
            if (focused_)
                focused_ = false;
            else
                engine.Quit();
            return;
        }
        if (!started_)
            Start();
        else if (waiting_)
            PollReply(engine);
        else if (ctx_.autoplay && frame_ - frameOfLastChange_ == kAutoDelay)
            AutoTurn(engine);
        else
            UpdateTurn(b);
    }

    // The comparison over the whole screen: the question and the answer field, then a row per buyer.
    void Render(sa::Engine& engine) override
    {
        sa::Backend& b = engine.GetBackend();
        b.DrawRect({0, 0}, {kWidth, kHeight}, ui::kBackground);
        RenderHeader(b);
        if (state_.is_object())
            RenderGrid(b);
        for (const Action& a : Actions())
            ui::DrawButton(b, a.button, mouse_.Position(), 1.6f);
        if (Asking())
            RenderField(b);
    }

private:
    // --- talking to the server ------------------------------------------------------------------

    // The comparison sale, once the client is free (a drawing screen's live guess may still be in flight).
    void Start()
    {
        if (ctx_.client.Busy())
        {
            ctx_.client.Poll();
            return;
        }
        started_ = true;
        nlohmann::json compare = nlohmann::json::array();
        for (const auto& m : models_)
            compare.push_back(m);
        Send({{"type", "round_start"}, {"session", ctx_.NewSession()}, {"item", "cat"}, {"buyers", 4},
              {"compare", compare}});
        status_ = "Every buyer is greeting you...";
    }

    void Send(nlohmann::json msg)
    {
        if (msg["type"] != "round_start")
            msg["session"] = ctx_.session;
        ctx_.client.Send(std::move(msg));
        waiting_ = true;
        focused_ = false;
    }

    // The server's reply: the comparison's new state (showing the newest turn), or an error.
    void PollReply(sa::Engine& engine)
    {
        const auto reply = ctx_.client.Poll();
        if (!reply)
            return;
        waiting_ = false;
        frameOfLastChange_ = frame_;
        if (reply->value("type", "") != "round_state")
        {
            status_ = ReplyError(ctx_, engine, *reply);
            return;
        }
        state_ = (*reply)["state"];
        status_.clear();
        input_.clear();
        view_ = Turns();
        if (ctx_.autoplay)
            for (const auto& line : (*reply)["new_lines"])
                std::cout << line.get<std::string>() << '\n';
        if (Finished())
            ctx_.sold = true;       // the comparison ran to the end (autoplay's pass)
    }

    // --- the player's turn ----------------------------------------------------------------------

    size_t Turns() const { return state_.is_object() ? state_["turns"].size() : 0; }
    bool Finished() const { return state_.is_object() && state_.value("finished", false); }
    bool Asking() const { return state_.is_object() && !Finished() && !waiting_; }

    // The player's input: focusing the field, typing (Enter answers), and the buttons.
    void UpdateTurn(sa::Backend& b)
    {
        if (mouse_.JustPressed())
            focused_ = Asking() && Field().Contains(mouse_.Position());
        else if (Asking() && !focused_ && b.KeyPressed(sa::Key::Enter))
        {
            focused_ = true;
            return;
        }
        if (ReadTyping(b))
        {
            Answer();
            return;
        }
        for (const Action& a : Actions())
            if (mouse_.Clicked(a.button))
            {
                Act(a.id);
                return;
            }
    }

    // A button: answer, page through the turns, or start the next sale.
    void Act(const std::string& id)
    {
        if (id == "answer")
            Answer();
        else if (id == "earlier")
            --view_;
        else if (id == "later")
            ++view_;
        else if (id == "next")
        {
            state_ = nlohmann::json();
            started_ = false;
        }
    }

    // Sends the typed answer to every model or way (nothing when the field is empty).
    void Answer()
    {
        if (input_.find_first_not_of(' ') == std::string::npos)
        {
            status_ = "Type your answer first.";
            return;
        }
        Send({{"type", "round_answer"}, {"text", input_}});
        status_ = "Every buyer is reacting...";
    }

    // Autoplay answers four questions the same way every time, then quits.
    void AutoTurn(sa::Engine& engine)
    {
        if (Finished() || autoTurns_ == 4)
        {
            engine.Quit();
            return;
        }
        input_ = kAutoAnswers[autoTurns_++];
        Answer();
    }

    // Adds this frame's typing to the field; true when Enter was pressed on something to send.
    bool ReadTyping(sa::Backend& b)
    {
        if (!focused_ || !Asking())
            return false;
        for (char c : b.TypedText())
            if (input_.size() < kMaxTyped)
                input_ += c;
        if (b.KeyPressed(sa::Key::Backspace) && !input_.empty())
            input_.pop_back();
        return b.KeyPressed(sa::Key::Enter) && input_.find_first_not_of(' ') != std::string::npos;
    }

    ui::Button Field() const { return {{kPad, kFieldTop}, {kFieldW, kFieldH}, ""}; }

    // Answer (while a question waits), turn paging, and the next sale (once this one is over).
    std::vector<Action> Actions() const
    {
        std::vector<Action> out;
        if (!state_.is_object())
            return out;
        const float x = kPad + kFieldW + 10, w = kWidth - kPad - x;
        if (Asking())
            out.push_back({"answer", {{x, kFieldTop}, {w, kFieldH}, "Answer", true, ui::ButtonStyle::Green}});
        else if (Finished())
            out.push_back({"next", {{x, kFieldTop}, {w, kFieldH}, "Next sale", !waiting_, ui::ButtonStyle::Green}});
        out.push_back({"earlier", {{kWidth - kPad - 250, 12}, {120, 34}, "< Earlier", view_ > 0 && !waiting_,
                                   ui::ButtonStyle::Cream}});
        out.push_back({"later", {{kWidth - kPad - 124, 12}, {120, 34}, "Later >", view_ < Turns() && !waiting_,
                                 ui::ButtonStyle::Cream}});
        return out;
    }

    // --- drawing --------------------------------------------------------------------------------

    std::string ModelName(size_t i) const { return i < models_.size() ? models_[i] : "?"; }

    // A column's heading: the way's label from the server ("top-k 50 (the game)"), or the model's name.
    std::string Label(size_t i) const
    {
        const std::string name = ModelName(i);
        if (state_.is_object() && state_.contains("labels") && state_["labels"].contains(name))
            return name.substr(0, name.find(':')) + ": " + state_["labels"][name].get<std::string>();
        return name;
    }

    // With more than two columns the buyer's column and the lines' text are smaller, so each line still fits.
    bool Many() const { return models_.size() > 2; }
    float InfoW() const { return Many() ? kInfoWide : kInfoW; }
    float LineScale() const { return Many() ? 1.3f : kTextScale; }
    float InfoScale() const { return Many() ? 1.25f : kInfoScale; }

    // The title, the turn in view (its question and your answer: what the rows below react to), and the question
    // that waits for your answer.
    void RenderHeader(sa::Backend& b) const
    {
        std::string title = "Comparing buyers:";
        for (size_t i = 0; i < models_.size(); ++i)
            title += (i ? " | " : " ") + ModelName(i);
        if (state_.is_object())
            title += "    sale seed " + std::to_string(state_.value("seed", 0)) + ", " +
                     (view_ == 0 ? std::string("openings") : "turn " + std::to_string(view_)) + " of " +
                     std::to_string(Turns());
        sa::DrawText(b, {kPad, 14}, title, 1.8f, ui::kInk);
        if (!state_.is_object())
        {
            sa::DrawText(b, {kPad, kLookTop}, status_, 1.6f, ui::kAccent);
            return;
        }
        std::string look = "Openings: each buyer greets you and makes a first offer.";
        if (view_ > 0 && view_ <= Turns())
        {
            const auto& t = state_["turns"][view_ - 1];
            look = "Turn " + std::to_string(view_) + ": " + t["asker"].get<std::string>() + " asked (written by " +
                   t["by"].get<std::string>() + "): " + t["text"].get<std::string>() + "  You: " +
                   t["answer"].get<std::string>();
        }
        DrawLines(b, look, kLookTop, 2, 1.45f, ui::kInk);
        std::string now;
        if (state_["question"].is_object())
        {
            const auto& q = state_["question"];
            now = "Now: " + q["asker"].get<std::string>() + " asks (written by " + q["by"].get<std::string>() +
                  "): " + q["text"].get<std::string>();
        }
        else
            now = "Everyone has asked. Saved: " + state_.value("saved", std::string());
        DrawLines(b, now, kNowTop, 2, 1.5f, ui::kGood);
        const std::string note = !status_.empty() ? status_
                                 : view_ < Turns() ? "Looking back: Later > returns to the newest turn."
                                                   : std::string();
        sa::DrawText(b, {kPad, kFieldTop + kFieldH + 4}, note, 1.3f, status_.empty() ? ui::kMuted : ui::kAccent);
    }

    void DrawLines(sa::Backend& b, const std::string& text, float top, size_t lines, float scale, sa::Color color) const
    {
        for (const auto& line : WrapLines(text, kWidth - 2 * kPad - 270, lines, scale))
        {
            sa::DrawText(b, {kPad, top}, line, scale, color);
            top += sa::LineHeight(scale);
        }
    }

    // The answer field, with a blinking cursor while it has focus.
    void RenderField(sa::Backend& b) const
    {
        const ui::Button f = Field();
        b.DrawRect({f.pos.x - 1, f.pos.y - 1}, {f.size.x + 2, f.size.y + 2}, ui::kBorder);
        b.DrawRect(f.pos, f.size, ui::kWhite);
        const sa::Vec2 text{f.pos.x + kFieldPad, f.pos.y + (kFieldH - sa::kGlyphSize * kFieldScale) / 2};
        std::string shown = input_.empty() && !focused_ ? std::string("Click here (or Enter) and type your answer")
                                                        : input_;
        while (!shown.empty() && sa::TextWidth(shown + "_", kFieldScale) > f.size.x - 2 * kFieldPad)
            shown.erase(0, 1);
        sa::DrawText(b, text, shown, kFieldScale, input_.empty() ? ui::kMuted : ui::kInk);
        if (focused_ && (frame_ / 30) % 2 == 0)
            sa::DrawText(b, {text.x + (input_.empty() ? 0 : sa::TextWidth(shown, kFieldScale)), text.y}, "_",
                         kFieldScale, ui::kAccent);
    }

    // A row per buyer: who they are, then each model's line for the turn in view (the openings at turn 0).
    void RenderGrid(sa::Backend& b) const
    {
        const auto& buyers = state_["buyers"];
        const size_t n = std::max<size_t>(1, models_.size());
        const float colW = (kWidth - 2 * kPad - InfoW()) / n;
        const float rowH = (kGridBottom - kGridTop - kHeadH) / std::max<size_t>(1, buyers.size());
        sa::DrawText(b, {kPad, kGridTop}, "Buyer", 1.5f, ui::kMuted);
        for (size_t m = 0; m < n; ++m)
            sa::DrawText(b, {kPad + InfoW() + m * colW + 8, kGridTop}, Label(m), Many() ? 1.3f : 1.6f, ui::kAccent);
        for (size_t i = 0; i < buyers.size(); ++i)
        {
            const float top = kGridTop + kHeadH + i * rowH;
            b.DrawRect({kPad, top}, {kWidth - 2 * kPad, 1}, ui::kBorder);
            RenderBuyer(b, buyers[i], {kPad, top + 6}, rowH - 10);
            for (size_t m = 0; m < n; ++m)
                RenderLine(b, buyers[i]["name"], ModelName(m), {kPad + InfoW() + m * colW + 8, top + 6},
                           {colW - 16, rowH - 10});
        }
    }

    // Who the buyer is: name, persona and what they care about.
    void RenderBuyer(sa::Backend& b, const nlohmann::json& buyer, sa::Vec2 at, float height) const
    {
        sa::DrawText(b, at, buyer["name"].get<std::string>(), 1.8f, ui::kInk);
        float y = at.y + sa::LineHeight(1.8f);
        std::string persona = buyer.value("persona", "");
        if (const size_t who = persona.find(", who "); who != std::string::npos)
            persona = persona.substr(who + 6);
        std::string cares;
        for (const auto& t : buyer["traits"])
            cares += (cares.empty() ? "cares about: " : ", ") + t.get<std::string>();
        const float bottom = at.y + height;
        for (const auto& [text, color] : {std::pair{persona, ui::kInk}, std::pair{cares, ui::kMuted}})
            for (const auto& line : WrapLines(text, InfoW() - 16, 4, InfoScale()))
            {
                if (y + sa::LineHeight(InfoScale()) > bottom)
                    return;
                sa::DrawText(b, {at.x, y}, line, InfoScale(), color);
                y += sa::LineHeight(InfoScale());
            }
    }

    // What one model's buyer did in the turn in view: a heading (the offer and its move, or why there is no line),
    // its colour, and what they said.
    struct Shown
    {
        std::string head;
        sa::Color color = ui::kInk;
        std::string says;
    };

    nlohmann::json Reaction(const std::string& name, const std::string& model) const
    {
        if (view_ == 0)
        {
            const auto& o = state_["openings"][model][name];
            return o["says"].is_null() ? nlohmann::json{{"readable", false}}
                                       : nlohmann::json{{"says", o["says"]}, {"offer", o["offer"]}};
        }
        return view_ <= Turns() ? state_["turns"][view_ - 1]["reactions"][model].value(name, nlohmann::json())
                                : nlohmann::json();
    }

    static Shown Describe(const nlohmann::json& r)
    {
        if (r.is_null())
            return {"(no line)", ui::kInk, ""};
        if (r.value("left_before", false))
            return {"(had left)", ui::kInk, ""};
        if (!r.value("readable", true))
            return {"unreadable", ui::kBad, r.value("says", std::string())};
        const int offer = r["offer"].is_number() ? r["offer"].get<int>() : 0;
        Shown shown{offer == 0 ? std::string("walks away") : Money(offer, false), ui::kInk, r.value("says", "")};
        if (!r.contains("old"))
            return shown;                   // an opening: the first offer, no move
        const int move = offer - r["old"].get<int>();
        if (move == 0)
            shown.head += "  (same)";
        else
        {
            shown.head += std::string(move > 0 ? "  (+" : "  (-") + Money(std::abs(move), false) + ")";
            shown.color = move > 0 ? ui::kGood : ui::kBad;
        }
        return shown;
    }

    // One model's line for one buyer in the turn in view, in a cell `size` big at `at`.
    void RenderLine(sa::Backend& b, const std::string& name, const std::string& model, sa::Vec2 at,
                    sa::Vec2 size) const
    {
        const Shown shown = Describe(Reaction(name, model));
        const float headScale = Many() ? 1.45f : 1.7f;
        sa::DrawText(b, at, shown.head, headScale, shown.color);
        float y = at.y + sa::LineHeight(headScale) + 2;
        const size_t room = static_cast<size_t>(std::max(0.0f, (at.y + size.y - y) / sa::LineHeight(LineScale())));
        for (const auto& line : WrapLines(shown.says, size.x, room, LineScale()))
        {
            sa::DrawText(b, {at.x, y}, line, LineScale(), ui::kInk);
            y += sa::LineHeight(LineScale());
        }
    }

    GameContext& ctx_;
    std::vector<std::string> models_;
    ui::Mouse mouse_;
    nlohmann::json state_;
    bool started_ = false;
    bool waiting_ = false;
    bool focused_ = false;
    std::string input_;
    std::string status_;
    size_t view_ = 0;               // 0: the openings; k: turn k
    int frame_ = 0;
    int frameOfLastChange_ = 0;
    int autoTurns_ = 0;
};

} // namespace

std::unique_ptr<sa::Scene> MakeCompareScene(GameContext& ctx, std::string models)
{
    return std::make_unique<CompareScene>(ctx, std::move(models));
}
