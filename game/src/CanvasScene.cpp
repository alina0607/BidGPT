#include "Scenes.h"
#include "Ui.h"

#include "sa/Text.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <iostream>
#include <string>

namespace
{

constexpr float kMinSpacing = 2.0f;
constexpr float kAiSize = 440.0f;           // the AI's drawing fits a square this big, centred
constexpr float kAiStep = 6.0f;             // the AI's pen moves this far per point (px)
constexpr int kAiPointsPerFrame = 3;        // ~1,000 px/s at 60 frames per second
constexpr int kAiPauseFrames = 8;           // a short pause between strokes, like lifting a pen
constexpr size_t kAiMaxTyped = 40;
constexpr float kFieldScale = 2.0f;

// A bicycle, so --autoplay has something to sell.
std::vector<Stroke> DemoBicycle()
{
    std::vector<Stroke> s;
    auto circle = [&](float cx, float cy, float r) {
        Stroke c;
        for (int i = 0; i <= 40; ++i)
        {
            const float a = 6.2831853f * i / 40;
            c.push_back({cx + r * std::cos(a), cy + r * std::sin(a)});
        }
        s.push_back(c);
    };
    circle(500, 430, 80);
    circle(780, 430, 80);
    s.push_back({{500, 430}, {600, 300}, {720, 300}, {780, 430}});
    s.push_back({{600, 300}, {650, 430}, {720, 300}});
    s.push_back({{650, 430}, {500, 430}});
    s.push_back({{720, 300}, {700, 250}, {740, 240}});
    s.push_back({{585, 290}, {625, 290}});
    return s;
}

// "a bicycle", "an apple"
std::string WithArticle(const std::string& noun)
{
    if (noun.empty())
        return noun;
    const char c = static_cast<char>(std::tolower(static_cast<unsigned char>(noun[0])));
    return (std::string("aeiou").find(c) != std::string::npos ? "an " : "a ") + noun;
}

class CanvasScene final : public sa::Scene
{
public:
    explicit CanvasScene(GameContext& ctx) : ctx_(ctx) {}

    void OnEnter(sa::Engine&) override { ClearDrawing(); }

    void Update(sa::Engine& engine, double dt) override
    {
        sa::Backend& b = engine.GetBackend();
        stage::Load(b);
        mouse_.Update(b);
        // The doodle board's exam form takes every click and key while it is open (Escape closes it, not the game).
        const bool form = ctx_.doodle.FormOpen();
        if (mouse_.JustPressed() && !form)
            aiFocused_ = AiField().Contains(mouse_.Position());
        ctx_.girl.Update(b, static_cast<float>(dt), !aiFocused_ && !form);
        ctx_.shelter.Update(b, static_cast<float>(dt));
        ctx_.doodle.Update(b, mouse_, static_cast<float>(dt));
        if (stage::LookingAtShelter() && ctx_.girl.X() != shelterGirlX_)
            stage::LookAtShelter(false);    // she walked off: the camera follows her again
        UpdateZoom(b, static_cast<float>(dt));
        if (stage::LookingAtDoodle() && ctx_.girl.X() != doodleGirlX_)
            stage::LookAtDoodle(false);
        ++frame_;
        if (form || ctx_.doodle.FormOpen())
            return;
        if (b.KeyPressed(sa::Key::Escape) && zoomCat_ < 0)   // with a painting enlarged, Escape only shrinks it
        {
            if (aiFocused_)
                aiFocused_ = false;         // Escape leaves the field before it leaves the game
            else
                engine.Quit();
            return;
        }
        if (waiting_)
        {
            PollClassification(engine);
            return;
        }
        PollReplies(engine);
        AnimateAi();
        if (mouse_.Clicked(PauseButton()))
        {
            ctx_.shelter.SetPaused(!ctx_.shelter.WantPaused());
            return;
        }
        if (zoomCat_ >= 0)
            return;                         // the enlarged painting takes the other clicks (UpdateZoom)
        if (mouse_.Clicked(BuyButton()))
        {
            engine.SetScene(MakeBuyScene(ctx_));
            return;
        }
        if (mouse_.Clicked(ShelterButton()))
            ToggleShelter(b);
        else if (mouse_.Clicked(ResetButton()))
        {
            ctx_.shelter.Start(b);
            ShowShelter();
        }
        else if (mouse_.JustPressed() && stage::EaselAt(mouse_.Position()) >= 0)
            OpenZoom(stage::EaselAt(mouse_.Position()));
        else if (mouse_.Clicked(DoodleButton()))
            ToggleDoodle();
        else if (UpdateAiField(b) || mouse_.Clicked(AiButton()) || AutoplayAsksAi())
            RequestAiDrawing();
        else if (mouse_.Clicked(SellButton()) || AutoplaySells())
            SendDrawing();
        else if (mouse_.Clicked(ClearButton()) || (!aiFocused_ && b.KeyPressed(sa::Key::C)))
            ClearDrawing();
        else if (!Animating())
            TrackPen(b);
    }

    void Render(sa::Engine& engine) override
    {
        sa::Backend& b = engine.GetBackend();
        stage::DrawWorld(b);
        stage::DrawOnCloth(b, ctx_.drawing, ui::kInk);
        ctx_.doodle.DrawFrame(b);
        ctx_.girl.Draw(b);
        ctx_.doodle.Draw(b, mouse_.Position());     // in front of her, so she never hides the paper
        const bool shelter = stage::LookingAtShelter();
        const bool doodle = !shelter && ctx_.doodle.InView();
        ui::Title(b, shelter  ? "The cat shelter: click a painting to see it big"
                     : doodle ? "彩色貓塗鴉：畫一隻貓，看 AI 讀出牠的樣子"
                              : "Draw on the cloth something you want to sell");
        ui::DrawButton(b, ShelterButton(), mouse_.Position());
        ui::DrawButton(b, BuyButton(), mouse_.Position());
        if (ctx_.shelter.Started())
            ui::DrawButton(b, ResetButton(), mouse_.Position());
        ui::DrawButton(b, DoodleButton(), mouse_.Position());
        ui::Panel(b, {20, 644}, {890, 66});
        ui::DrawButton(b, ClearButton(), mouse_.Position());
        ui::DrawButton(b, SellButton(), mouse_.Position());
        RenderAiField(b);
        ui::DrawButton(b, AiButton(), mouse_.Position());
        const std::string guess = liveGuess_.empty() ? "" : "I think it's: " + liveGuess_;
        constexpr float statusX = 70.0f;   // clear the flower cluster at the left of the panel
        if (shelter && !waiting_)
            sa::DrawText(b, {statusX, 668},
                         ctx_.shelter.Started() ? ctx_.shelter.Status() : "", 1.75f,
                         ctx_.shelter.Done() ? ui::kGood : ui::kAccent);
        else if (waiting_)
            sa::DrawText(b, {statusX, 668}, "Looking at your drawing...", 2.0f, ui::kMuted);
        else if (doodle)
            sa::DrawText(b, {statusX, 666}, "選一個顏色和一支筆，在布幕上畫一隻貓；右邊會即時顯示 AI 讀到什麼", 1.8f,
                         ui::kMuted);
        else if (!error_.empty())
            sa::DrawWrapped(b, {statusX, 652}, error_, 820, 1.5f, ui::kBad);
        else if (awaitingAi_ || aiQueued_)
            sa::DrawText(b, {statusX, 668}, "The AI is thinking how to draw it...", 2.0f, ui::kMuted);
        else if (Animating())
            sa::DrawText(b, {statusX, 668}, "The AI is drawing " + WithArticle(aiLabel_) + ".  " + guess, 2.0f,
                         ui::kAccent);
        else if (!guess.empty() && !ctx_.drawing.empty())
            sa::DrawText(b, {statusX, 668},
                         (aiLabel_.empty() ? "" : "The AI drew " + WithArticle(aiLabel_) + ".  ") + guess, 2.0f,
                         ui::kAccent);
        else
            sa::DrawText(b, {statusX, 668}, "WASD/arrows move | Space jump | Shift slide | J kick | K heart | Drag draw | C clear", 1.5f, ui::kMuted);
        ctx_.doodle.DrawForm(b, mouse_.Position());
        RenderZoom(b);
        if (ctx_.shelter.CanPause())        // over the enlarged painting too
            ui::DrawButton(b, PauseButton(), mouse_.Position());
    }

private:
    // A live guess may still be in flight when "Sell it!" is pressed (the client takes one request
    // at a time), so the final request is queued until that reply is in, and partial replies are
    // never taken for the final answer.
    void PollClassification(sa::Engine& engine)
    {
        if (finalQueued_)
        {
            if (ctx_.client.Busy())
            {
                ctx_.client.Poll();          // drop the live guess's reply once it arrives
                return;
            }
            finalQueued_ = false;
            ctx_.client.Send(DrawingRequest(false));
            return;
        }
        const auto reply = ctx_.client.Poll();
        if (!reply || reply->value("partial", false))
            return;
        waiting_ = false;
        if ((*reply)["type"] != "classify_result")
        {
            error_ = ReplyError(ctx_, engine, *reply);
            return;
        }
        ctx_.candidates.clear();
        for (const auto& c : (*reply)["top_k"])
            ctx_.candidates.push_back({c["label"], c["prob"]});
        engine.SetScene(MakeConfirmScene(ctx_));
    }

    // Autoplay asks the AI for a bicycle at frame 10 (live guesses follow each of its strokes) and
    // sells the drawing once it is finished. Right before selling it asks for one more live guess,
    // so the final request has to wait behind it (the queued path). Returns true on the frame it sells.
    bool AutoplayAsksAi()
    {
        if (!ctx_.autoplay || frame_ != 10)
            return false;
        aiInput_ = "bicycle";
        return true;
    }

    bool AutoplaySells()
    {
        if (!ctx_.autoplay || frame_ <= 10 || AiBusy() || ctx_.drawing.empty())
            return false;
        RequestLiveGuess();
        return true;
    }

    // The drawing as a classify request; `partial` marks a live guess.
    nlohmann::json DrawingRequest(bool partial) const
    {
        nlohmann::json strokes = nlohmann::json::array();
        for (const Stroke& s : ctx_.drawing)
        {
            nlohmann::json xs = nlohmann::json::array(), ys = nlohmann::json::array();
            for (sa::Vec2 p : s)
            {
                xs.push_back(std::round(p.x));
                ys.push_back(std::round(p.y));
            }
            strokes.push_back({xs, ys});
        }
        nlohmann::json request = {{"type", "classify"}, {"strokes", strokes}};
        if (partial)
            request["partial"] = true;
        return request;
    }

    void SendDrawing()
    {
        stage::LookAtShelter(false);
        finalQueued_ = ctx_.client.Busy();
        if (!finalQueued_)
            ctx_.client.Send(DrawingRequest(false));
        waiting_ = true;
        error_.clear();
    }

    // Live guessing: after every finished stroke, ask what the drawing looks like so far. One
    // request at a time; strokes drawn meanwhile are covered by the next request. A guess that
    // arrives after the canvas was cleared is dropped.
    void RequestLiveGuess()
    {
        if (ctx_.client.Busy() || ctx_.drawing.empty())
        {
            liveDirty_ = !ctx_.drawing.empty();
            return;
        }
        liveDirty_ = false;
        liveGeneration_ = generation_;
        ctx_.client.Send(DrawingRequest(true));
    }

    // Takes whatever reply came in (a live guess, if it is still for this drawing, or the AI's
    // drawing), then sends what is waiting once the client is free: the AI request first, then a
    // live guess for strokes finished in the meantime.
    void PollReplies(sa::Engine& engine)
    {
        const auto reply = ctx_.client.Poll();
        if (reply && awaitingAi_ && !reply->value("partial", false))
            TakeAiReply(engine, *reply);
        else if (reply && reply->value("partial", false) && liveGeneration_ == generation_ &&
                 (*reply)["type"] == "classify_result" && !(*reply)["top_k"].empty())
        {
            const auto& top = (*reply)["top_k"][0];
            liveGuess_ = top.value("label", "") + " (" +
                         std::to_string(static_cast<int>(std::round(top.value("prob", 0.0) * 100))) + "%)";
            if (ctx_.autoplay)
                std::cout << "autoplay: live guess " << liveGuess_ << '\n';
        }
        if (aiQueued_ && !ctx_.client.Busy())
            SendAiRequest();
        else if (liveDirty_ && !ctx_.client.Busy() && !drawing_)
            RequestLiveGuess();
    }

    // --- "AI draw": the drawing GPT draws what was typed (or a surprise) on the canvas ---------

    // Typing goes to the field only while it has focus (click it); Enter asks for the drawing.
    bool UpdateAiField(sa::Backend& b)
    {
        if (mouse_.JustPressed())
            aiFocused_ = AiField().Contains(mouse_.Position());
        if (!aiFocused_)
            return false;
        for (char c : b.TypedText())
            if (aiInput_.size() < kAiMaxTyped)
                aiInput_ += c;
        if (b.KeyPressed(sa::Key::Backspace) && !aiInput_.empty())
            aiInput_.pop_back();
        return b.KeyPressed(sa::Key::Enter) && AiButton().enabled;
    }

    void RequestAiDrawing()
    {
        stage::LookAtShelter(false);        // the AI draws on the cloth: show it
        aiFocused_ = false;
        error_.clear();
        aiGeneration_ = generation_;
        aiQueued_ = ctx_.client.Busy();     // one request at a time: wait behind a live guess
        if (!aiQueued_)
            SendAiRequest();
    }

    void SendAiRequest()
    {
        aiQueued_ = false;
        awaitingAi_ = true;
        ctx_.client.Send({{"type", "draw"}, {"text", aiInput_}});
    }

    void TakeAiReply(sa::Engine& engine, const nlohmann::json& reply)
    {
        awaitingAi_ = false;
        if (aiGeneration_ != generation_)
            return;                         // the canvas was cleared meanwhile
        if (reply.value("type", "") == "draw_result" && reply.value("ok", false))
        {
            StartAiAnimation(reply["strokes"], reply.value("label", ""));
            return;
        }
        if (ctx_.autoplay)                  // a test against a server without a drawing model
        {
            std::cout << "autoplay: no AI drawing (" << reply.value("message", "?") << "), drawing a bicycle\n";
            ctx_.drawing = DemoBicycle();
            return;
        }
        error_ = reply.value("type", "") == "draw_result" ? reply.value("message", "")
                                                           : ReplyError(ctx_, engine, reply);
    }

    // The strokes arrive in 0..255; they are scaled into a square in the middle of the canvas and
    // resampled every kAiStep px, so the pen moves at an even speed whatever the stroke's shape.
    void StartAiAnimation(const nlohmann::json& strokes, const std::string& label)
    {
        ClearDrawing();
        aiGeneration_ = generation_;
        aiLabel_ = label;
        const float scale = kAiSize / 255.0f;
        const sa::Vec2 origin{stage::kCanvasLeft + (stage::kCanvasWidth - kAiSize) / 2,
                              stage::kCanvasTop + (stage::CanvasHeight() - kAiSize) / 2};
        aiStrokes_.clear();
        for (const auto& s : strokes)
        {
            Stroke target;
            for (size_t i = 0; i < s[0].size(); ++i)
            {
                const sa::Vec2 p{origin.x + s[0][i].get<float>() * scale, origin.y + s[1][i].get<float>() * scale};
                if (!target.empty())
                {
                    const sa::Vec2 d = p - target.back();
                    const int steps = static_cast<int>(std::sqrt(d.x * d.x + d.y * d.y) / kAiStep);
                    const sa::Vec2 from = target.back();
                    for (int k = 1; k < steps; ++k)
                        target.push_back({from.x + d.x * k / steps, from.y + d.y * k / steps});
                }
                target.push_back(p);
            }
            if (!target.empty())
                aiStrokes_.push_back(target);
        }
        aiStroke_ = aiPoint_ = 0;
        aiPause_ = 0;
        if (ctx_.autoplay)
            std::cout << "autoplay: the AI draws " << WithArticle(label) << " (" << aiStrokes_.size() << " strokes)\n";
    }

    bool Animating() const { return aiStroke_ < aiStrokes_.size(); }

    // A few points per frame; at the end of each stroke a short pause and a live guess, so the
    // classifier guesses the AI's drawing as it grows, the same way it guesses the player's.
    void AnimateAi()
    {
        if (!Animating())
            return;
        if (aiPause_ > 0)
        {
            --aiPause_;
            return;
        }
        const Stroke& target = aiStrokes_[aiStroke_];
        if (aiPoint_ == 0)
            ctx_.drawing.push_back({});
        for (int k = 0; k < kAiPointsPerFrame && aiPoint_ < target.size(); ++k)
            ctx_.drawing.back().push_back(target[aiPoint_++]);
        if (aiPoint_ < target.size())
            return;
        ++aiStroke_;
        aiPoint_ = 0;
        aiPause_ = kAiPauseFrames;
        RequestLiveGuess();
    }

    void RenderAiField(sa::Backend& b) const
    {
        const ui::Button f = AiField();
        ui::Panel(b, f.pos, f.size);
        const float leftPad = 38.0f;   // the field skin's flower sits inside its left edge
        const float rightPad = 24.0f;
        const sa::Vec2 text{f.pos.x + leftPad, f.pos.y + (f.size.y - sa::kGlyphSize * kFieldScale) / 2};
        if (aiInput_.empty() && !aiFocused_)
        {
            sa::DrawText(b, {text.x, f.pos.y + (f.size.y - sa::kGlyphSize * 1.5f) / 2}, "What should the AI draw?",
                         1.5f, ui::kMuted);
            return;
        }
        // The end of what was typed, if it is wider than the field.
        std::string shown = aiInput_;
        const float cursorW = sa::TextWidth("_", kFieldScale);
        while (!shown.empty() && sa::TextWidth(shown, kFieldScale) + cursorW > f.size.x - leftPad - rightPad)
            shown.erase(0, 1);
        sa::DrawText(b, text, shown, kFieldScale, ui::kInk);
        if (aiFocused_ && (frame_ / 30) % 2 == 0)
            sa::DrawText(b, {text.x + sa::TextWidth(shown, kFieldScale), text.y}, "_", kFieldScale, ui::kAccent);
    }

    void ClearDrawing()
    {
        ctx_.drawing.clear();
        liveGuess_.clear();
        liveDirty_ = false;
        aiStrokes_.clear();
        aiStroke_ = aiPoint_ = 0;
        aiLabel_.clear();
        ++generation_;
    }

    // Pressing on the cloth starts a stroke; dragging extends it (points closer than kMinSpacing
    // canvas units are skipped); a stroke started on the cloth may continue off it.
    void TrackPen(sa::Backend& b)
    {
        const bool inCanvas = stage::OnCloth(mouse_.Position());
        const sa::Vec2 m = stage::ScreenToCanvas(mouse_.Position());
        const bool down = b.MouseDown(sa::MouseButton::Left);
        if (down && inCanvas && !drawing_)
            ctx_.drawing.push_back({m});
        else if (down && drawing_ && !ctx_.drawing.empty())
        {
            const sa::Vec2 d = m - ctx_.drawing.back().back();
            if (std::sqrt(d.x * d.x + d.y * d.y) >= kMinSpacing)
                ctx_.drawing.back().push_back(m);
        }
        const bool wasDrawing = drawing_;
        drawing_ = down && (inCanvas || drawing_);
        if (wasDrawing && !drawing_)
            RequestLiveGuess();             // a stroke just ended
    }

    // --- The cat shelter: Brush GPT paints its eight easels (Shelter.h) -------------------------

    // A painting clicked on its easel grows into a big canvas in the middle of the screen. It is the easel's own
    // texture, so the strokes the AI is still laying keep appearing on it, with rings on the last ones (the fine
    // hairs are too small to see land otherwise). While the eye model paints that cat's eyes, the view zooms on
    // into the eye canvas, the eye model's own view, and back out when they are done. Any click or Escape shrinks
    // it back onto its easel.
    static constexpr float kZoomSeconds = 0.45f;
    static constexpr float kEyeZoomSeconds = 0.9f;
    static constexpr float kZoomSide = 540.0f;      // screen px of the big canvas
    static constexpr float kZoomTop = 72.0f;

    void OpenZoom(int cat)
    {
        zoomCat_ = cat;
        zoomOpening_ = true;
    }

    void UpdateZoom(sa::Backend& b, float dt)
    {
        if (zoomCat_ < 0)
            return;
        const bool pauseClick = mouse_.Clicked(PauseButton());     // pausing keeps the painting big
        if (zoomOpening_ && ((mouse_.JustPressed() && !pauseClick) || b.KeyPressed(sa::Key::Escape) ||
                             !stage::EaselPainting(zoomCat_)))
            zoomOpening_ = false;           // the click that closes it does nothing else
        zoomT_ = std::clamp(zoomT_ + (zoomOpening_ ? dt : -dt) / kZoomSeconds, 0.0f, 1.0f);
        const bool eyes = zoomCat_ == ctx_.shelter.PaintingCat() && ctx_.shelter.PaintingEyes() &&
                          ctx_.shelter.EyeCanvas(zoomCat_);
        eyeT_ = std::clamp(eyeT_ + (eyes ? dt : -dt) / kEyeZoomSeconds, 0.0f, 1.0f);
        if (!zoomOpening_ && zoomT_ <= 0.0f)
        {
            zoomCat_ = -1;
            eyeT_ = 0.0f;
        }
    }

    void RenderZoom(sa::Backend& b) const
    {
        const sa::TextureId painting = stage::EaselPainting(zoomCat_);
        if (zoomCat_ < 0 || !painting)
            return;
        const float t = zoomT_ * zoomT_ * (3.0f - 2.0f * zoomT_);           // ease in and out
        sa::Vec2 from, fromSize;
        stage::EaselScreenRect(zoomCat_, from, fromSize);
        const sa::Vec2 to{(1280.0f - kZoomSide) / 2, kZoomTop};
        const sa::Vec2 pos{from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t};
        const sa::Vec2 size{fromSize.x + (kZoomSide - fromSize.x) * t, fromSize.y + (kZoomSide - fromSize.y) * t};
        b.DrawRect({0, 0}, {1280, 720}, {0.18f, 0.12f, 0.10f, 0.6f * t});
        constexpr float kMat = 10.0f;
        b.DrawRect({pos.x - kMat, pos.y - kMat}, {size.x + 2 * kMat, size.y + 2 * kMat}, {1.0f, 0.99f, 0.97f, 1.0f});
        // Zooming into the eye canvas: the face's eye box grows to fill the frame, then the eye canvas fades in.
        const float e = eyeT_ * eyeT_ * (3.0f - 2.0f * eyeT_);
        const std::array<float, 4> box = ctx_.shelter.EyeBox(zoomCat_);
        constexpr float kSide = 256.0f;      // the server's canvases
        const sa::UVRect uv{{box[0] / kSide * e, box[1] / kSide * e},
                            {1.0f + ((box[2] - box[0]) / kSide - 1.0f) * e, 1.0f + ((box[3] - box[1]) / kSide - 1.0f) * e}};
        b.DrawSprite(painting, uv, pos, size);
        const sa::TextureId eyeCanvas = ctx_.shelter.EyeCanvas(zoomCat_);
        if (eyeCanvas && e > 0.6f)
            b.DrawSprite(eyeCanvas, {}, pos, size, (e - 0.6f) / 0.4f);
        if (t < 1.0f)
            return;
        // Rings on the last strokes (newest brightest), on whichever canvas is filling the frame.
        if (zoomCat_ == ctx_.shelter.PaintingCat() && (e <= 0.0f || e >= 1.0f) && ctx_.shelter.PaintingEyes() == (e >= 1.0f))
        {
            const auto& recent = ctx_.shelter.Recent();
            for (size_t i = 0; i < recent.size(); ++i)
            {
                const float age = static_cast<float>(recent.size() - 1 - i) / std::max<size_t>(1, recent.size());
                const sa::Vec2 c{pos.x + recent[i].x / kSide * size.x, pos.y + recent[i].y / kSide * size.y};
                const float r = i + 1 == recent.size() ? 9.0f : 6.0f;
                const sa::Color ring{1.0f, 0.25f + 0.55f * age, 0.45f, 1.0f - 0.85f * age};
                for (int k = 0; k < 12; ++k)
                {
                    const float a0 = k * 6.2831853f / 12, a1 = (k + 1) * 6.2831853f / 12;
                    b.DrawLine({c.x + r * std::cos(a0), c.y + r * std::sin(a0)}, {c.x + r * std::cos(a1), c.y + r * std::sin(a1)},
                               i + 1 == recent.size() ? 2.5f : 1.5f, ring);
                }
            }
        }
        const std::string line = ctx_.shelter.CatStatus(zoomCat_);
        const float y = kZoomTop + kZoomSide + kMat + 12;
        sa::DrawText(b, {(1280.0f - sa::TextWidth(line, 1.75f)) / 2, y}, line, 1.75f, {1, 1, 1, 1});
        const std::string hint = "Click anywhere to put it back";
        sa::DrawText(b, {(1280.0f - sa::TextWidth(hint, 1.4f)) / 2, y + 30}, hint, 1.4f, {0.92f, 0.86f, 0.82f, 1});
    }

    // The first press paints the eight cats; later presses only move the camera between shelter and stall.
    void ToggleShelter(sa::Backend& b)
    {
        if (stage::LookingAtShelter())
        {
            stage::LookAtShelter(false);
            return;
        }
        if (!ctx_.shelter.Started())
            ctx_.shelter.Start(b);
        ShowShelter();
    }

    void ShowShelter()
    {
        stage::LookAtShelter(true);
        shelterGirlX_ = ctx_.girl.X();
    }

    // The colour cat doodle station (DoodleBoard.h) is also reached by walking left; this takes the camera there.
    void ToggleDoodle()
    {
        if (stage::LookingAtDoodle())
        {
            stage::LookAtDoodle(false);
            return;
        }
        stage::LookAtDoodle(true);
        doodleGirlX_ = ctx_.girl.X();
    }

    ui::Button ShelterButton() const
    {
        return {{14, 82}, {200, 44}, stage::LookingAtShelter() ? "Back to stall" : "Cat shelter", !waiting_,
                ui::ButtonStyle::Cream};
    }
    // Buying a cat: the model is the seller and the other buyers (BuyScene).
    ui::Button BuyButton() const
    {
        return {{14, 134}, {200, 44}, "Buy a cat", !waiting_ && !AiBusy(), ui::ButtonStyle::Green};
    }
    // Holds the AI's painting (and its clock) so a stretch that looks wrong can be named by its seconds.
    ui::Button PauseButton() const
    {
        return {{1060, 120}, {200, 44}, ctx_.shelter.WantPaused() ? "Resume painting" : "Pause painting",
                !waiting_ && ctx_.shelter.CanPause(), ui::ButtonStyle::Pink};
    }
    ui::Button ResetButton() const
    {
        return {{224, 82}, {120, 44}, "Reset", !waiting_ && ctx_.shelter.Started(), ui::ButtonStyle::Pink};
    }

    ui::Button DoodleButton() const
    {
        // Under "AI draw", clear of the stall's sign (the main one and the doodle station's copy).
        return {{1060, 66}, {200, 44}, stage::LookingAtDoodle() ? "回到攤位" : "彩色貓塗鴉", !waiting_,
                ui::ButtonStyle::Green};
    }

    ui::Button ClearButton() const
    {
        return {{930, 652}, {150, 50}, "Clear", !waiting_, ui::ButtonStyle::Cream};
    }
    ui::Button SellButton() const
    {
        return {{1095, 652}, {165, 50}, "Sell it!", !waiting_ && !ctx_.drawing.empty() && !AiBusy(),
                ui::ButtonStyle::Pink};
    }
    // Empty text draws a surprise; otherwise the server matches it to one of the 345 drawable things.
    ui::Button AiField() const { return {{790, 14}, {290, 44}, "", true}; }
    ui::Button AiButton() const
    {
        return {{1095, 14}, {165, 44}, "AI draw", !waiting_ && !AiBusy(), ui::ButtonStyle::Green};
    }
    bool AiBusy() const { return awaitingAi_ || aiQueued_ || Animating(); }

    GameContext& ctx_;
    ui::Mouse mouse_;
    bool drawing_ = false;
    bool waiting_ = false;
    bool finalQueued_ = false;      // "Sell it!" pressed while a live guess was in flight
    bool liveDirty_ = false;        // strokes finished since the last live-guess request
    int generation_ = 0;            // bumped on every clear
    int liveGeneration_ = -1;       // generation the in-flight live guess was made for
    std::string liveGuess_;
    std::string error_;
    std::string aiInput_;           // what the player asked the AI to draw
    bool aiFocused_ = false;
    bool aiQueued_ = false;         // "AI draw" pressed while a live guess was in flight
    bool awaitingAi_ = false;       // the draw request is out
    int aiGeneration_ = -1;         // generation the AI drawing was asked for
    std::string aiLabel_;           // what the AI drew (shown with the live guess)
    std::vector<Stroke> aiStrokes_; // the AI's strokes in canvas pixels, still being drawn
    size_t aiStroke_ = 0, aiPoint_ = 0;
    int aiPause_ = 0;
    int frame_ = 0;
    float shelterGirlX_ = 0.0f;     // where she stood when the camera went to the shelter
    int zoomCat_ = -1;              // the shelter painting shown big (-1: none)
    bool zoomOpening_ = false;      // growing or shown (false: shrinking back)
    float zoomT_ = 0.0f;            // 0 on its easel .. 1 big
    float eyeT_ = 0.0f;             // 0 the whole face .. 1 the eye canvas
    float doodleGirlX_ = 0.0f;      // where she stood when the camera went to the doodle station
};

} // namespace

std::unique_ptr<sa::Scene> MakeCanvasScene(GameContext& ctx) { return std::make_unique<CanvasScene>(ctx); }

std::string ReplyError(GameContext& ctx, sa::Engine& engine, const nlohmann::json& reply)
{
    std::string message = reply.value("message", "Something went wrong.");
    if (ctx.autoplay)
    {
        std::cerr << "autoplay: server error: " << message << '\n';
        engine.Quit();
    }
    return message;
}

void DrawStrokesInBox(sa::Backend& b, const std::vector<Stroke>& strokes, sa::Vec2 pos, sa::Vec2 size,
                      float thickness, sa::Color color)
{
    float minX = 1e9f, minY = 1e9f, maxX = -1e9f, maxY = -1e9f;
    for (const Stroke& s : strokes)
        for (sa::Vec2 p : s)
        {
            minX = std::min(minX, p.x), maxX = std::max(maxX, p.x);
            minY = std::min(minY, p.y), maxY = std::max(maxY, p.y);
        }
    if (minX > maxX)
        return;
    const float pad = 20.0f;
    const float scale = std::min((size.x - 2 * pad) / std::max(1.0f, maxX - minX),
                                 (size.y - 2 * pad) / std::max(1.0f, maxY - minY));
    const sa::Vec2 offset{pos.x + (size.x - (maxX - minX) * scale) / 2,
                          pos.y + (size.y - (maxY - minY) * scale) / 2};
    auto map = [&](sa::Vec2 p) { return sa::Vec2{offset.x + (p.x - minX) * scale, offset.y + (p.y - minY) * scale}; };
    for (const Stroke& s : strokes)
    {
        if (s.size() == 1)
            b.DrawLine(map(s[0]), map(s[0]), thickness, color);
        for (size_t i = 1; i < s.size(); ++i)
            b.DrawLine(map(s[i - 1]), map(s[i]), thickness, color);
    }
}
