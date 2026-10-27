#pragma once

#include "MlClient.h"
#include "Ui.h"

#include "sa/Backend.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <vector>

// The colour cat doodle station, left of the stall (stage::kDoodleX): a sheet of beige paper the player paints a
// cat on with a 10-colour palette and four brushes, and beside it, live, what the models read from the picture
// (docs/protocol.md, read_doodle): whether the sketch classifier sees a cat, and the coat, pattern, patches and
// eyes the cat-doodle-attributes reader gives. The board only shows what the models say: whatever the reader
// leaves out (everything, while no reader is installed) is shown as unspecified. Further right (stage::kFrameX)
// stands the empty frame Brush GPT will later paint the described cat in.
//
// Only the pencil strokes (the outline) go to the classifier: it learnt from Quick, Draw!'s black lines and never
// sees colour, so the painted fill would read as stripes. The whole picture goes to the reader.
//
// "存下塗鴉（當考卷）" keeps the doodle as an exam for the reader (docs/protocol.md, save_doodle_exam): a form,
// prefilled with what the reader says now, where the owner corrects every field to the true answer; the server
// stores the picture, the strokes, the answer, which fields were changed and the reader's own prediction.
//
// The board is a copy of the stall, and its cloth is the canvas, the same square cloth the main game is drawn on
// (owner, 2026-10-17; scripts/make_stage_art.py STALL_STRETCH). The paint is a picture kept in memory (kResW × kResH
// RGBA, transparent where nothing is painted): every stroke is painted into it on the CPU. On screen it lies on the
// cloth; the reader gets the same paint on the beige paper its training doodles are drawn on (the cloth's colour is
// close to it), reduced to 256 × 256. It has its own connection to the server, like the shelter, so a reading never waits behind the cloth's
// guesses.
class DoodleBoard
{
public:
    static constexpr int kResW = 768, kResH = 768;  // the paint in memory: the cloth is square
    static constexpr int kSent = 256;               // the picture read_doodle gets (a third of kResW)

    enum class Tool
    {
        Pencil,     // a thin line
        Flat,       // the oil painter's flat brush: a rounded rectangle with faint bristle streaks
        Taper,      // a pointed brush: thin at both ends of the stroke
        Eraser,
    };

    DoodleBoard(std::string host, int port);

    // Tools, the pen and the reading; call every frame.
    void Update(sa::Backend& backend, const ui::Mouse& mouse, float dt);
    void DrawFrame(sa::Backend& backend) const;               // the empty frame (behind the girl)
    void Draw(sa::Backend& backend, sa::Vec2 mouse) const;    // easel, paper, tools and panel (in front of her)
    // A press here is the station's (the screen's own buttons stay clear of it).
    bool Contains(sa::Vec2 screen) const;
    // On screen enough to take the title and the hint line.
    bool InView() const;
    void Clear();
    // The exam form is open: it takes every click and key until it is saved or cancelled.
    bool FormOpen() const { return form_.open; }
    void DrawForm(sa::Backend& backend, sa::Vec2 mouse) const;     // on top of everything, screen space

private:
    struct Stroke
    {
        Tool tool = Tool::Pencil;
        int colour = 0;             // index into the palette
        bool thick = false;
        std::vector<sa::Vec2> points;   // paint pixels, 0..kResW × 0..kResH
    };

    // What the server said about the doodle, for the panel.
    struct Reading
    {
        bool have = false;
        bool isCat = false;
        std::string top;            // the classifier's first two guesses with their probabilities
        nlohmann::json attributes;
        std::string reader;
    };

    // One answer on the exam form: a drop-down of names (the first is unspecified, null).
    struct Option
    {
        std::string code;           // empty: unspecified
        std::string zh;
    };
    struct FormField
    {
        std::string label;
        std::string key;            // coat, pattern, left_eye, right_eye, or a patch region
        bool patch = false;
        std::vector<Option> options;
        int prefill = 0;            // what the reader said when the form opened
        int choice = 0;
    };
    struct Form
    {
        bool open = false;
        std::vector<FormField> fields;
        int dropdown = -1;          // the field whose list is open
    };

    void TakeClick(sa::Vec2 station);
    bool Current() const;           // the reading shown is for the drawing as it is now
    bool HasPencil() const;
    void OpenForm();
    void UpdateForm(sa::Backend& backend, const ui::Mouse& mouse);
    void SendExam();
    nlohmann::json Strokes(bool pencilOnly) const;
    std::vector<std::uint8_t> Picture() const;     // the reader's kSent × kSent RGBA
    nlohmann::json Image() const;
    void TrackPen(const ui::Mouse& mouse, bool down);
    void EndStroke();
    void Poll();
    void SendReading();
    nlohmann::json Request() const;

    void Paint(const Stroke& stroke, std::vector<std::uint8_t>& rgba);
    void Show(sa::Backend& backend);
    void EraseInk(const Stroke& eraser);

    void DrawTools(sa::Backend& b, sa::Vec2 o, sa::Vec2 mouse) const;
    void DrawPanel(sa::Backend& b, sa::Vec2 o, sa::Vec2 mouse) const;

    MlClient client_;
    Tool tool_ = Tool::Pencil;
    int colour_ = 0;
    bool thick_ = false;
    std::vector<std::uint8_t> paper_;   // every finished stroke (straight alpha; 0 where the cloth shows)
    std::vector<std::uint8_t> shown_;   // the paper plus the stroke being drawn
    std::vector<float> cover_, phase_;  // one stroke's coverage and where across the brush each pixel is
    Stroke current_;
    bool penDown_ = false;
    std::vector<Stroke> ink_;           // what is drawn and not erased (the eraser cuts strokes)
    std::vector<Stroke> history_;       // every stroke as drawn, eraser included, for the exam
    bool anything_ = false;             // something was painted since the last clear (erased or not)
    sa::TextureId texture_ = 0;
    sa::TextureId preview_ = 0;         // the reader's picture, shown on the exam form
    bool previewDirty_ = false;
    bool changed_ = true;               // shown_ needs to go to the texture
    bool dirty_ = false;                // strokes since the last request
    float sinceStroke_ = 0.0f;
    int generation_ = 0;                // bumped on every clear
    int requested_ = -1;                // generation of the request in flight
    Reading reading_;
    std::string error_;
    enum class Awaiting
    {
        Nothing,
        Reading,
        Exam,
    };
    Awaiting awaiting_ = Awaiting::Nothing;
    Form form_;
    bool examQueued_ = false;           // confirmed while a reading was still out
    nlohmann::json exam_;               // the exam waiting to be sent
    std::string examStatus_;
    bool examFailed_ = false;
};
