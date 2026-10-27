// The girl's controls on the headless backend with the real art: run while ← or → is held, stop when let
// go, walk up and down the screen with ↑ ↓, jump on Space and land again; and a painted buyer's walk.

#include "Girl.h"
#include "Npc.h"
#include "Stage.h"

#include <cstdio>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <string>

namespace
{

int failures = 0;

void Check(bool ok, const char* what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    failures += ok ? 0 : 1;
}

// The headless backend for textures, with keys the test holds down or presses.
struct Keys final : sa::Backend
{
    std::unique_ptr<sa::Backend> null = sa::CreateBackend("null");
    bool left = false, right = false, up = false, down = false, space = false;
    bool w = false, a = false, s = false, d = false, shift = false, j = false, k = false;
    std::map<sa::TextureId, std::string> textures;
    std::string drawn;
    sa::UVRect uv{};
    sa::Vec2 drawnSize{};
    bool heartDrawn = false;
    sa::Vec2 heartPosition{};
    sa::UVRect heartUV{};
    float heartAlpha = 1.0f;
    struct Draw { sa::Vec2 pos, size; };
    bool recordDraws = false;
    std::vector<Draw> draws;

    std::string_view Name() const override { return "keys"; }
    bool Init(const sa::WindowDesc& d) override { return null->Init(d); }
    void Shutdown() override {}
    bool BeginFrame(sa::Color) override { return true; }
    void EndFrame() override {}
    sa::Vec2 MousePosition() const override { return {}; }
    bool MouseDown(sa::MouseButton) const override { return false; }
    bool KeyPressed(sa::Key key) const override
    {
        return (key == sa::Key::Space && space) || (key == sa::Key::Shift && shift) ||
               (key == sa::Key::J && j) || (key == sa::Key::K && k);
    }
    bool KeyDown(sa::Key key) const override
    {
        return (key == sa::Key::Left && left) || (key == sa::Key::Right && right) || (key == sa::Key::Up && up) ||
               (key == sa::Key::Down && down) || (key == sa::Key::W && w) || (key == sa::Key::A && a) ||
               (key == sa::Key::S && s) || (key == sa::Key::D && d);
    }
    void DrawLine(sa::Vec2, sa::Vec2, float, sa::Color) override {}
    void DrawRect(sa::Vec2, sa::Vec2, sa::Color) override {}
    sa::TextureId LoadTexture(const std::string& path, sa::TextureFilter filter) override
    {
        const auto texture = null->LoadTexture(path, filter);
        textures[texture] = path;
        return texture;
    }
    sa::Vec2 TextureSize(sa::TextureId t) const override { return null->TextureSize(t); }
    void DrawSprite(sa::TextureId t, sa::UVRect source, sa::Vec2 pos, sa::Vec2 size, float alpha) override
    {
        drawn = textures[t];
        uv = source;
        drawnSize = size;
        if (recordDraws) draws.push_back({pos, size});
        if (drawn.find("heart_wave.png") != std::string::npos)
        {
            heartDrawn = true;
            heartPosition = pos;
            heartUV = source;
            heartAlpha = alpha;
        }
    }
};

void Run(Girl& girl, Keys& keys, float seconds)
{
    for (float t = 0; t < seconds; t += 1.0f / 60)
    {
        girl.Update(keys, 1.0f / 60);
        keys.space = false;   // a press lasts one frame
        keys.shift = keys.j = keys.k = false;
    }
}

} // namespace

int main()
{
    Keys keys;
    Girl girl;
    Run(girl, keys, 0.5f);
    const float start = girl.X();
    Check(start == stage::GirlStart(), "she starts where the layout says");

    keys.right = true;
    Run(girl, keys, 2.0f);
    const float ran = girl.X() - start;
    Check(ran >= 515 && ran <= 530, "holding right for 2 s runs at 260 px/s without a fake turn delay");

    keys.right = false;
    Run(girl, keys, 1.5f);
    const float stopped = girl.X();
    Run(girl, keys, 1.0f);
    Check(stopped - start - ran < 60 && girl.X() == stopped, "letting go she slides a little and stands still");

    keys.left = true;
    Run(girl, keys, 1.0f);
    Check(girl.X() < stopped, "holding ← runs her left");
    keys.left = false;

    const float ground = girl.Y();
    keys.up = true;
    Run(girl, keys, 2.0f);   // 0.9 s turning her back to the camera, then 1.1 s at 140 px/s
    keys.up = false;
    Check(girl.Y() < ground - 100, "holding ↑ walks her up the screen (after turning her back to the camera)");
    keys.down = true;
    Run(girl, keys, 2.0f);
    keys.down = false;
    Check(girl.Y() > ground - 100, "holding ↓ walks her back down (after turning round)");

    keys.space = true;
    float apex = 0.0f;
    for (int i = 0; i < 60; ++i)
    {
        girl.Update(keys, 1.0f / 60);
        keys.space = false;
        apex = std::max(apex, girl.Lift());
    }
    Check(apex > 322 && apex < 325, "Space apex is twice the previous physical height (323.5 px)");
    Run(girl, keys, 1.0f);
    Check(girl.Lift() == 0.0f, "and she lands again");
    keys.space = true;
    girl.Update(keys, 1.0f / 60, false);
    Check(girl.Lift() == 0.0f, "not while a text field takes the Space key");
    keys.space = false;

    // Letter aliases behave like arrows, including turn animations for depth movement.
    Girl letters, arrows;
    for (int direction = 0; direction < 4; ++direction)
    {
        for (int frame = 0; frame < 120; ++frame)
        {
            keys.d = direction == 0; keys.a = direction == 1;
            keys.w = direction == 2; keys.s = direction == 3;
            letters.Update(keys, 1.0f / 60);
            keys.d = keys.a = keys.w = keys.s = false;
            keys.right = direction == 0; keys.left = direction == 1;
            keys.up = direction == 2; keys.down = direction == 3;
            arrows.Update(keys, 1.0f / 60);
            keys.right = keys.left = keys.up = keys.down = false;
        }
        Check(std::abs(letters.X() - arrows.X()) < .001f && std::abs(letters.Y() - arrows.Y()) < .001f,
              "WASD matches the corresponding arrow exactly");
    }

    for (int fps : {30, 60, 144})
    {
        Girl jumping;
        float peak = 0;
        std::set<int> poses;
        for (int frame = 0; frame < fps * 2; ++frame)
        {
            keys.space = frame == 0;
            jumping.Update(keys, 1.0f / fps);
            jumping.Draw(keys);
            peak = std::max(peak, jumping.Lift());
            if (keys.drawn.find("girl_jump.png") != std::string::npos)
                poses.insert(int(std::round(keys.uv.offset.x * 8)));
        }
        Check(std::abs(peak - 2 * 784.0f * 784.0f / (2 * 1900)) < .3f,
              "double-height apex is stable at 30/60/144 fps");
        Check(poses == std::set<int>({1, 2, 3, 4, 5, 6, 7}), "jump renders crouch, launch, rise, apex, fall, landing and recovery");
        Check(jumping.Lift() == 0 && jumping.CurrentAction() == Girl::Action::None, "jump finishes grounded and idle");
    }
    keys.space = false;
    for (bool kick : {false, true})
    {
        for (bool left : {false, true})
        {
            Girl action;
            Run(action, keys, .2f);
            const float before = action.X();
            std::set<int> poses;
            bool mirror = true;
            for (int frame = 0; frame < 120; ++frame)
            {
                keys.j = kick && frame == 0;
                keys.shift = !kick && frame == 0;
                keys.a = left && frame == 0;
                action.Update(keys, 1.0f / 60);
                action.Draw(keys);
                if (keys.drawn.find(kick ? "girl_kick.png" : "girl_slide.png") != std::string::npos)
                {
                    poses.insert(int(std::round(keys.uv.offset.x * 8)) - (left ? 1 : 0));
                    mirror &= (keys.uv.size.x < 0) == left;
                }
            }
            Check(poses == std::set<int>({1, 2, 3, 4, 5, 6, 7}), "each one-shot action renders all seven active poses");
            Check(mirror, "slide/kick mirrors for the facing direction");
            Check(action.CurrentAction() == Girl::Action::None, "slide/kick returns to idle automatically");
            Check(kick ? action.X() == before : (action.X() - before) * (left ? -1 : 1) > 150,
                  "kick stays planted; slide travels forward");
        }
    }
    Girl typing;
    Run(typing, keys, .2f);
    const float typedX = typing.X(), typedY = typing.Y();
    keys.d = keys.w = keys.j = keys.k = keys.shift = keys.space = true;
    for (int i = 0; i < 120; ++i) typing.Update(keys, 1.0f / 60, false);
    Check(typing.X() == typedX && typing.Y() == typedY && typing.Lift() == 0 && typing.CurrentAction() == Girl::Action::None,
          "text focus blocks WASD, jump, slide, kick and heart cast");
    keys.d = keys.w = keys.j = keys.k = keys.shift = keys.space = false;

    for (bool left : {false, true})
    {
        Girl caster;
        Run(caster, keys, .5f);
        const float standing = caster.X();
        std::set<int> poses;
        size_t maximumHearts = 0;
        float firstX = 0, lastX = 0, minimumAlpha = 1;
        bool seen = false, mirrored = true, premature = false;
        for (int frame = 0; frame < 150; ++frame)
        {
            keys.k = frame == 0;
            keys.a = left && frame == 0;
            caster.Update(keys, 1.0f / 60);
            if (frame < 10) premature |= caster.HeartCount() != 0;
            maximumHearts = std::max(maximumHearts, caster.HeartCount());
            keys.heartDrawn = false;
            caster.Draw(keys);
            if (keys.drawn.find("girl_cast.png") != std::string::npos)
                poses.insert(int(std::round(keys.uv.offset.x * 6)) - (left ? 1 : 0));
            if (keys.heartDrawn)
            {
                if (!seen) firstX = keys.heartPosition.x;
                lastX = keys.heartPosition.x;
                seen = true;
                mirrored &= (keys.heartUV.size.x < 0) == left;
                minimumAlpha = std::min(minimumAlpha, keys.heartAlpha);
            }
        }
        Check(!premature && maximumHearts == 1, "K launches exactly one wave after the hand windup");
        Check(poses == std::set<int>({1, 2, 3, 4, 5}), "cast plays all five active hand poses");
        Check(seen && mirrored && (lastX - firstX) * (left ? -1 : 1) > 400,
              "heart texture flies and mirrors in the facing direction");
        Check(minimumAlpha < .1f && caster.HeartCount() == 0, "wave fades and is removed after its lifetime");
        Check(caster.CurrentAction() == Girl::Action::None && caster.X() == standing,
              "casting stays planted and returns to idle");
    }

    // Inspect rendered textures/UVs, not just Action::None: recovery is still an animation.
    // Rejected turn art must never be activated merely because it was generated.
    for (bool left : {false,true})
    {
        Girl whole;
        whole.Update(keys,0);
        keys.recordDraws=true;
        bool intact=true, rejectedAbsent=true;
        for (int frame=0;frame<180;++frame)
        {
            keys.a=left && frame<90;
            keys.d=!left && frame<90;
            whole.Update(keys,1.0f/60);
            keys.draws.clear();
            whole.Draw(keys);
            intact &= keys.draws.size()==1;
            rejectedAbsent &= keys.drawn.find("girl_transition.png")==std::string::npos;
        }
        Check(intact, "character is one complete sprite, never horizontal bands or two-pose dissolves");
        Check(rejectedAbsent, "rejected v2/v3 turn is never drawn");
        Check(keys.drawn.find("girl_idle.png")!=std::string::npos,"stop finishes in original idle");
        keys.recordDraws=false;
        keys.a=keys.d=false;
    }
    for (int phase=0;phase<8;++phase)
    {
        Girl settle;
        keys.d=true;
        settle.Update(keys,0);
        Run(settle,keys,phase*.12f);
        keys.d=false;
        const float stoppedX=settle.X();
        Run(settle,keys,.75f);
        settle.Draw(keys);
        Check(keys.drawn.find("girl_idle.png")!=std::string::npos && settle.X()-stoppedX<30,
              "release at each gait phase settles to idle with bounded braking");
    }
    for (bool away : {false, true})
    {
        Girl walk;
        Run(walk, keys, .2f);
        keys.w = away; keys.s = !away;
        Run(walk, keys, away ? 1.35f : .45f);
        keys.w = keys.s = false;
        const float stoppedY = walk.Y();
        std::set<int> poses;
        for (int frame = 0; frame < 18; ++frame)
        {
            walk.Update(keys, 1.0f / 60);
            walk.Draw(keys);
            if (keys.drawn.find("girl_walk.png") != std::string::npos)
                poses.insert(int(std::round(keys.uv.offset.x * 30)));
        }
        Check(poses.size() >= 2, "vertical walking finishes its half-step before idle");
        Run(walk, keys, .5f);
        walk.Draw(keys);
        Check(walk.Y() == stoppedY && (away ? std::abs(keys.uv.offset.x) < .001f :
              keys.drawn.find("girl_idle.png") != std::string::npos), "walk return stays in place and finishes with planted feet");
    }
    Girl tap;
    keys.d=true;
    Run(tap,keys,.2f);
    keys.d=false;
    Run(tap,keys,.05f);
    tap.Draw(keys);
    Check(keys.drawn.find("girl_run.png")!=std::string::npos,"release completes the existing stride");
    keys.d=true;
    Run(tap,keys,1.1f);
    tap.Draw(keys);
    Check(keys.drawn.find("girl_run.png")!=std::string::npos,"movement can resume while settling");
    keys.d=false;

    Girl idle;
    idle.Update(keys,0);
    keys.recordDraws=true;
    keys.draws.clear();
    idle.Draw(keys);
    const auto firstIdle=keys.draws;
    idle.Update(keys,1.8f);
    keys.draws.clear();
    idle.Draw(keys);
    Check(firstIdle.size()==1 && keys.draws.size()==1, "idle is a whole authored frame");
    if(firstIdle.size()==1 && keys.draws.size()==1)
        Check(firstIdle[0].pos.x==keys.draws[0].pos.x && firstIdle[0].pos.y==keys.draws[0].pos.y &&
              firstIdle[0].size.x==keys.draws[0].size.x && firstIdle[0].size.y==keys.draws[0].size.y,
              "idle has a fixed root and scale; no procedural torso stretch");
    keys.recordDraws=false;

    // A painted buyer walks up, stands, and walks off when they lose interest.
    npcs::Load(keys);
    Check(npcs::Count() >= 7, "all seven NPC drawings load");
    Walker walker(0, 0.5f);
    for (int i = 0; i < 60 * 4; ++i)
        walker.Update(1.0f / 60, true);
    Check(walker.Settled() && !walker.Gone(), "an interested buyer walks up and stands at the stall");
    walker.Update(1.0f / 60, false);
    Check(!walker.Settled(), "losing interest, they turn away (their tag hides while they walk)");
    for (int i = 0; i < 60 * 5; ++i)
        walker.Update(1.0f / 60, false);
    Check(walker.Gone(), "and walk off out of the picture");

    std::printf(failures ? "%d failure(s)\n" : "all girl tests passed\n", failures);
    return failures ? 1 : 0;
}
