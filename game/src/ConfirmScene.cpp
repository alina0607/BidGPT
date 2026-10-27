#include "Scenes.h"
#include "Ui.h"

#include "sa/Text.h"

#include <string>
#include <vector>

namespace
{

class ConfirmScene final : public sa::Scene
{
public:
    explicit ConfirmScene(GameContext& ctx) : ctx_(ctx) {}

    void Update(sa::Engine& engine, double dt) override
    {
        sa::Backend& b = engine.GetBackend();
        ctx_.girl.Update(b, static_cast<float>(dt));
        ctx_.shelter.Update(b, static_cast<float>(dt));
        mouse_.Update(b);
        ++frame_;
        if (b.KeyPressed(sa::Key::Escape))
        {
            engine.Quit();
            return;
        }
        if (waiting_)
        {
            if (auto reply = ctx_.client.Poll())
            {
                waiting_ = false;
                if ((*reply)["type"] == "round_state")
                    engine.SetScene(MakeRoundScene(ctx_, *reply));
                else
                    error_ = ReplyError(ctx_, engine, *reply);
            }
            return;
        }
        const auto buttons = Buttons();
        for (size_t i = 0; i < ctx_.candidates.size(); ++i)
        {
            if (mouse_.Clicked(buttons[i]) || (ctx_.autoplay && i == 0 && frame_ == 45))
            {
                ctx_.client.Send({{"type", "round_start"},
                                  {"session", ctx_.NewSession()},
                                  {"item", ctx_.candidates[i].label},
                                  {"buyers", 4}});
                waiting_ = true;
                error_.clear();
                return;
            }
        }
        if (mouse_.Clicked(buttons.back()))
            engine.SetScene(MakeCanvasScene(ctx_));
    }

    void Render(sa::Engine& engine) override
    {
        sa::Backend& b = engine.GetBackend();
        stage::DrawWorld(b);
        stage::DrawOnCloth(b, ctx_.drawing, ui::kInk);
        ctx_.girl.Draw(b);
        ui::Title(b, "Is it one of these?");
        ui::Panel(b, {40, 100}, {440, 500}, ui::kWhite);
        DrawStrokesInBox(b, ctx_.drawing, {40, 100}, {440, 440}, 4.0f, ui::kInk);
        sa::DrawText(b, {60, 560}, "Your drawing", 2.0f, ui::kMuted);
        if (waiting_ || !error_.empty())
            ui::Panel(b, {520, 486}, {720, 64});

        const auto buttons = Buttons();
        for (const auto& button : buttons)
            ui::DrawButton(b, button, mouse_.Position(), 2.5f);
        if (waiting_)
            sa::DrawText(b, {540, 500}, "Calling the buyers over...", 2.0f, ui::kMuted);
        else if (!error_.empty())
            sa::DrawWrapped(b, {540, 500}, error_, 700, 1.5f, ui::kBad);
    }

private:
    std::vector<ui::Button> Buttons() const
    {
        std::vector<ui::Button> out;
        float y = 120;
        for (const Candidate& c : ctx_.candidates)
        {
            out.push_back({{540, y}, {680, 70},
                           c.label + "  (" + std::to_string(static_cast<int>(c.prob * 100 + 0.5f)) + "%)",
                           !waiting_});
            y += 90;
        }
        out.push_back({{540, y + 20}, {680, 60}, "None of these - draw again", !waiting_});
        return out;
    }

    GameContext& ctx_;
    ui::Mouse mouse_;
    bool waiting_ = false;
    std::string error_;
    int frame_ = 0;
};

} // namespace

std::unique_ptr<sa::Scene> MakeConfirmScene(GameContext& ctx) { return std::make_unique<ConfirmScene>(ctx); }
