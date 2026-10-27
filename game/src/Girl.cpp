#include "Girl.h"

#include "Characters.h"
#include "Sheets.h"
#include "Stage.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <string>
#include <utility>

namespace
{

constexpr float kRunSpeed = 260.0f;     // px/s sideways
constexpr float kWalkSpeed = 140.0f;    // px/s up and down the screen
// Height = v²/(2g): sqrt(2) times 784 doubles the previous height, not quadruples it.
constexpr float kJump = 784.0f * 1.41421356237f;
constexpr float kGravity = 1900.0f;     // px/s²
constexpr float kRiseOrFall = 200.0f;   // px/s: slower than this she is at the top of the jump
constexpr float kPreparation = 0.10f;
constexpr float kLanding = 0.32f;
constexpr float kCastRelease = 0.20f;   // windup + hand thrust, then launch on the open-palm pose
constexpr float kHeartSpeed = 550.0f;
constexpr float kHeartLife = 1.2f;
constexpr float kWalkSettle = 0.30f;

struct Art
{
    bool tried = false;
    Sheet idle, run, walk, jump, slide, kick, cast, transition;
    sa::TextureId heart = 0;
    sa::Vec2 heartSize{};
    std::map<Girl::Face, Sheet> faces;
};

Art art;

void Load(sa::Backend& backend)
{
    if (art.tried)
        return;
    art.tried = true;
    const std::string dir = AssetDir() + "/stage";
    art.idle = LoadSheet(backend, dir, "girl_idle");
    art.run = LoadSheet(backend, dir, "girl_run");
    art.transition = LoadSheet(backend, dir, "girl_turn_approved");
    art.walk = LoadSheet(backend, dir, "girl_walk");
    art.jump = LoadSheet(backend, dir, "girl_jump");
    art.slide = LoadSheet(backend, dir, "girl_slide");
    art.kick = LoadSheet(backend, dir, "girl_kick");
    art.cast = LoadSheet(backend, dir, "girl_cast");
    art.heart = backend.LoadTexture(dir + "/heart_wave.png", sa::TextureFilter::Smooth);
    art.heartSize = backend.TextureSize(art.heart);
    const std::pair<Girl::Face, const char*> faces[] = {
        {Girl::Face::Sheepish, "girl_sheepish"}, {Girl::Face::Angry, "girl_angry"}, {Girl::Face::Anxious, "girl_anxious"}};
    for (const auto& [face, name] : faces)
        if (std::ifstream(dir + "/" + name + ".json"))
            art.faces[face] = LoadSheet(backend, dir, name);
}

float PartLength(const Sheet& s, const char* part)
{
    const auto [first, last] = s.Part(part);
    return s.Length(first, last);
}

float TurnLength()
{
    return std::max(art.transition.Length(0, art.transition.sheet.frames-1), 0.01f);
}

// Frame-by-frame art is indivisible: never deform horizontal bands or dissolve two poses.
void DrawIdle(sa::Backend& backend, const Sheet& sheet, int frame, sa::Vec2 feet, float scale)
{
    DrawFrame(backend, sheet, frame, feet, scale);
}

void DrawTurn(sa::Backend& backend, sa::Vec2 feet, int heading, float progress, float idlePhase)
{
    // The replacement sheet is opt-in: rejected/generated drafts must never become runtime art.
    const float scale = stage::GirlHeight() / 320.0f;
    if (progress <= 0.0f || !art.transition.Loaded())
    {
        DrawFrame(backend, art.idle, art.idle.FrameAt(idlePhase, 0, art.idle.sheet.frames-1, true), feet, scale);
        return;
    }
    if (progress >= 1.0f)
    {
        DrawFrame(backend, art.run, 4, feet, scale, heading < 0);
        return;
    }
    const int frame = art.transition.FrameAt(progress * TurnLength(), 0, art.transition.sheet.frames-1, false);
    DrawFrame(backend, art.transition, frame, feet, scale, heading < 0 && frame != 0);
}
} // namespace

void Girl::Go(Motion motion)
{
    if (motion == Motion::Starting && motion_ == Motion::Standing)
        turnIdlePhase_ = time_;
    if (motion == Motion::Stopping && motion_ == Motion::Running)
    {
        runPhase_ = std::fmod(time_, art.run.Length(1,8));
        const float planted = art.run.Length(1,3);
        runStopAdvance_ = (runPhase_ < planted ? planted : art.run.Length(1,8)) - runPhase_;
    }
    if (motion_ != motion)
        time_ = 0.0f;
    motion_ = motion;
}

void Girl::Recover()
{
    action_ = Action::None;
    fromBehind_ = false;
    Go(art.transition.Loaded() ? Motion::Recovering : Motion::Standing);
    time_ = 0.0f;
}

void Girl::StopRun(float dt)
{
    const float length = (art.transition.Loaded() ? TurnLength() : 0.0f)+runStopAdvance_;
    // Pivoting poses have planted feet: brake briefly, not skating through the whole turn.
    const float braking = std::min(length, 0.18f);
    x_ += heading_ * kRunSpeed * 0.4f * std::max(0.0f, 1.0f - time_ / braking) * dt;
    if (time_ >= length)
        Go(Motion::Standing);
}

void Girl::StopWalk()
{
    const auto [first, last] = art.walk.Part(fromBehind_ ? "away" : "toward");
    const float halfStep = art.walk.Length(first, first + (last - first + 1) / 2 - 1);
    walkStopPhase_ = std::fmod(time_, halfStep * 2.0f);
    // Continue to the next planted pose, rather than snapping from a lifted foot to idle.
    const float phase = std::fmod(walkStopPhase_, halfStep);
    // The first drawing of either half-step already has both feet down.
    walkStopAdvance_ = phase < art.walk.seconds[first] ? 0.0f : halfStep - phase;
    Go(Motion::WalkStopping);
}

void Girl::Feel(Face face, float seconds)
{
    if (faceLeft_ > 0.0f && static_cast<int>(face) < static_cast<int>(face_))
        return;
    face_ = face;
    faceLeft_ = seconds;
}

void Girl::Update(sa::Backend& backend, float dt, bool controlsEnabled)
{
    stage::Load(backend);
    Load(backend);
    if (!placed_)
    {
        x_ = stage::GirlStart();
        y_ = stage::Ground();
        placed_ = true;
    }
    for (auto& wave : hearts_)
    {
        wave.age += dt;
        wave.x += wave.direction * kHeartSpeed * dt;
    }
    hearts_.erase(std::remove_if(hearts_.begin(), hearts_.end(), [](const HeartWave& wave) {
        return wave.age >= kHeartLife;
    }), hearts_.end());
    const auto held = [&](sa::Key key, sa::Key alias) {
        return controlsEnabled && (backend.KeyDown(key) || backend.KeyDown(alias));
    };
    const int across = int(held(sa::Key::Right, sa::Key::D)) - int(held(sa::Key::Left, sa::Key::A));
    const int down = int(held(sa::Key::Down, sa::Key::S)) - int(held(sa::Key::Up, sa::Key::W));
    if (controlsEnabled && action_ == Action::None)
    {
        if (backend.KeyPressed(sa::Key::Space))
            action_ = Action::JumpPreparation;
        else if (backend.KeyPressed(sa::Key::Shift))
            action_ = Action::Slide;
        else if (backend.KeyPressed(sa::Key::J))
            action_ = Action::Kick;
        else if (backend.KeyPressed(sa::Key::K))
            action_ = Action::Cast;
        if (action_ != Action::None)
        {
            actionTime_ = 0.0f;
            castReleased_ = false;
            if (across) heading_ = across;
            fromBehind_ = false;
            Go(Motion::Standing);
        }
    }
    actionTime_ += dt;
    float flightDt = dt;
    if (action_ == Action::JumpPreparation && actionTime_ >= kPreparation)
    {
        action_ = Action::Jump;
        actionTime_ -= kPreparation;
        flightDt = actionTime_;
        rise_ = kJump;
    }
    if (action_ == Action::Jump)
    {
        // Exact constant-acceleration integration makes the apex independent of frame rate.
        lift_ += rise_ * flightDt - 0.5f * kGravity * flightDt * flightDt;
        rise_ -= kGravity * flightDt;
        if (lift_ <= 0.0f && rise_ < 0.0f)
        {
            lift_ = rise_ = 0.0f;
            action_ = Action::Landing;
            actionTime_ = 0.0f;
        }
    }
    else if (action_ == Action::Landing && actionTime_ >= kLanding)
        Recover();
    time_ += dt;
    faceLeft_ = std::max(0.0f, faceLeft_ - dt);

    if (action_ == Action::Slide || action_ == Action::Kick || action_ == Action::Cast)
    {
        const Sheet& sheet = action_ == Action::Slide ? art.slide : action_ == Action::Kick ? art.kick : art.cast;
        const float length = sheet.Length(1, sheet.sheet.frames - 1);
        if (action_ == Action::Cast && !castReleased_ && actionTime_ >= kCastRelease)
        {
            castReleased_ = true;
            // Bound the live effect list even if a future controller adds rapid-fire input.
            if (hearts_.size() < 8)
            {
                const float overshoot = actionTime_ - kCastRelease;
                hearts_.push_back({x_ + heading_ * (stage::GirlHeight() * 0.46f + kHeartSpeed * overshoot),
                                   y_ - stage::GirlHeight() * 0.54f, overshoot, heading_});
            }
        }
        if (action_ == Action::Slide && actionTime_ > 0.10f && actionTime_ < 0.76f)
            x_ += heading_ * 420.0f * std::clamp((0.76f - actionTime_) / 0.28f, 0.0f, 1.0f) * dt;
        if (actionTime_ >= length)
            Recover();
        stage::Follow(x_, dt);
        return;
    }

    if (across != 0)
    {
        Run(across, dt);
        y_ += down * kWalkSpeed * dt;   // running diagonally
    }
    else if (down != 0)
        Walk(down, dt);
    else
        switch (motion_)
        {
        case Motion::Starting:
        {
            // A brief tap reverses the turn already in progress; it must not jump to a full stride.
            const float progress = std::clamp(time_ / TurnLength(), 0.0f, 1.0f);
            Go(Motion::Stopping);
            runStopAdvance_ = 0;
            time_ = (1.0f - progress)*TurnLength();
            break;
        }
        case Motion::Running:
            Go(Motion::Stopping);
            break;
        case Motion::Stopping:
            StopRun(dt);
            break;
        case Motion::Recovering:
            if (time_ >= TurnLength())
                Go(Motion::Standing);
            break;
        case Motion::WalkingAway:
        case Motion::WalkingToward:
            StopWalk();
            break;
        case Motion::WalkStopping:
            if (time_ >= kWalkSettle)
                Go(Motion::Standing);
            break;
        case Motion::Turning:
            if (time_ >= PartLength(art.walk, "turn"))
            {
                fromBehind_ = turningAway_;
                Go(Motion::Standing);
            }
            break;
        case Motion::Standing:
            break;
        }
    stage::Follow(x_, dt);
}

void Girl::Run(int way, float dt)
{
    // Preserve the rear→front turn when a vertical walk changes into a sideways run.
    if (motion_ == Motion::WalkingAway || (fromBehind_ && motion_ != Motion::Turning))
    {
        turningAway_ = false;
        Go(Motion::Turning);
    }
    if (motion_ == Motion::Turning)
    {
        if (turningAway_)
        {
            time_ = std::max(0.0f, PartLength(art.walk, "turn") - time_);
            turningAway_ = false;
        }
        if (time_ < PartLength(art.walk, "turn"))
            return;
        fromBehind_ = false;
        Go(Motion::Standing);
    }
    heading_ = way;
    fromBehind_ = false;
    if (motion_ == Motion::Standing)
    {
        Go(art.transition.Loaded() ? Motion::Starting : Motion::Running);
        if (motion_ == Motion::Running)
            time_ = art.run.Length(1,3);
    }
    else if (motion_ == Motion::Stopping || motion_ == Motion::Recovering)
    {
        if (!art.transition.Loaded())
        {
            const float resume = runPhase_ + std::min(time_, runStopAdvance_);
            Go(Motion::Running);
            time_ = resume;
            x_ += way * kRunSpeed * dt;
            return;
        }
        const bool recovering = motion_ == Motion::Recovering;
        const float progress = std::clamp((time_-(recovering?0:runStopAdvance_))/TurnLength(), 0.0f, 1.0f);
        Go(Motion::Starting);
        time_ = (1.0f - progress)*TurnLength();
    }
    else if (motion_ != Motion::Starting)
        Go(Motion::Running);
    if (motion_ == Motion::Starting)
    {
        const float length = TurnLength();
        const float launch = std::clamp((time_ / length - 0.6f) / 0.4f, 0.0f, 1.0f);
        x_ += way * kRunSpeed * 0.5f * launch * launch * (3.0f - 2.0f * launch) * dt;
        if (time_ >= length)
        {
            const float overshoot=time_-length;
            Go(Motion::Running);
            time_=art.run.Length(1,3)+overshoot;
        }
    }
    else
        x_ += way * kRunSpeed * dt;
}

void Girl::Walk(int way, float dt)
{
    const bool away = way < 0;
    if (motion_ == Motion::Turning)
    {
        if (time_ >= PartLength(art.walk, "turn"))
        {
            fromBehind_ = turningAway_;
            Go(fromBehind_ ? Motion::WalkingAway : Motion::WalkingToward);
        }
        return;
    }
    const bool behindNow = fromBehind_;
    if (away != behindNow)
    {
        turningAway_ = away;
        Go(Motion::Turning);
        return;
    }
    fromBehind_ = away;
    Go(away ? Motion::WalkingAway : Motion::WalkingToward);
    y_ += way * kWalkSpeed * dt;
}

void Girl::Draw(sa::Backend& backend) const
{
    if (art.heart && art.heartSize.x > 0)
        for (const auto& wave : hearts_)
        {
            const sa::Vec2 center = stage::ToScreen({wave.x, wave.y});
            const float width = 155.0f * (1.0f + 0.035f * std::sin(wave.age * 18.0f));
            const float height = width * art.heartSize.y / art.heartSize.x;
            const bool mirrored = wave.direction < 0;
            const sa::UVRect uv = mirrored ? sa::UVRect{{1, 0}, {-1, 1}} : sa::UVRect{};
            const float opacity = std::clamp((kHeartLife - wave.age) / 0.25f, 0.0f, 1.0f);
            // Reference point is the leading heart, not the trailing ribbon's center.
            backend.DrawSprite(art.heart, uv, {center.x - width * (mirrored ? .28f : .72f), center.y - height * .5f},
                               {width, height}, opacity);
        }
    const sa::Vec2 feet = stage::ToScreen({x_, y_ - lift_});
    // Every direction and action shares one exact on-screen height.  Perspective scaling made the
    // walk/run sheets visibly shrink before snapping larger again when idle resumed.
    auto scale = [&](const Sheet& s) { return stage::GirlHeight() / s.standing; };
    if (action_ != Action::None)
    {
        const Sheet& sheet = action_ == Action::Slide ? art.slide : action_ == Action::Kick ? art.kick :
                             action_ == Action::Cast ? art.cast : art.jump;
        int frame = 1;
        if (action_ == Action::Slide || action_ == Action::Kick || action_ == Action::Cast)
            frame = sheet.FrameAt(actionTime_, 1, sheet.sheet.frames - 1, false);
        else if (action_ == Action::Landing)
            frame = actionTime_ < 0.10f ? 6 : 7;
        else if (action_ == Action::Jump)
            frame = actionTime_ < 0.09f ? 2 : rise_ > kRiseOrFall ? 3 : rise_ < -kRiseOrFall ? 5 : 4;
        DrawFrame(backend, sheet, frame, feet, scale(sheet), heading_ < 0);
        return;
    }
    switch (motion_)
    {
    case Motion::Standing:
    {
        if (fromBehind_)
        {
            DrawIdle(backend, art.walk, art.walk.Part("away").first, feet, scale(art.walk));
            return;
        }
        const auto face = art.faces.find(face_);
        if (faceLeft_ > 0.0f && face != art.faces.end() && face->second.Loaded())
            DrawIdle(backend, face->second, 0, feet, scale(face->second));
        else
            DrawIdle(backend, art.idle, art.idle.FrameAt(time_, 0, art.idle.sheet.frames - 1, true), feet,
                     scale(art.idle));
        return;
    }
    case Motion::Starting:
    case Motion::Stopping:
    case Motion::Recovering:
    {
        if(motion_==Motion::Stopping && time_<runStopAdvance_) {
            DrawFrame(backend,art.run,art.run.FrameAt(runPhase_+time_,1,9,false),feet,scale(art.run),heading_<0);
            return;
        }
        const float progress=motion_==Motion::Starting ? time_/TurnLength() :
            1-(time_-(motion_==Motion::Stopping?runStopAdvance_:0))/TurnLength();
        DrawTurn(backend,feet,heading_,progress,motion_==Motion::Starting?turnIdlePhase_:0);
        return;
    }
    case Motion::Running:
    {
        const auto [first, last] = art.run.Part("run");
        DrawFrame(backend, art.run, art.run.FrameAt(time_, first, last, true), feet, scale(art.run), heading_ < 0);
        return;
    }
    case Motion::WalkStopping:
    {
        const auto [first, last] = art.walk.Part(fromBehind_ ? "away" : "toward");
        const float phase = walkStopPhase_ + walkStopAdvance_ * std::clamp(time_ / kWalkSettle, 0.0f, 1.0f);
        DrawFrame(backend, art.walk, art.walk.FrameAt(phase, first, last, true), feet, scale(art.walk));
        return;
    }
    case Motion::WalkingAway:
    case Motion::WalkingToward:
    {
        const auto [first, last] = art.walk.Part(motion_ == Motion::WalkingAway ? "away" : "toward");
        DrawFrame(backend, art.walk, art.walk.FrameAt(time_, first, last, true), feet, scale(art.walk));
        return;
    }
    case Motion::Turning:
    {
        // The drawings turn from behind to facing the camera; turning the other way plays them backwards.
        const auto [first, last] = art.walk.Part("turn");
        const int frame = art.walk.FrameAt(time_, first, last, false);
        DrawFrame(backend, art.walk, turningAway_ ? last - (frame - first) : frame, feet, scale(art.walk));
        return;
    }
    }
}
