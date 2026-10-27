#pragma once

#include "sa/Backend.h"

#include <vector>

// WASD/arrows move, Space jumps, Shift slides, J spin-kicks, K casts a heart wave.
class Girl
{
public:
    enum class Face
    {
        None,
        Sheepish,   // the player has just answered and the buyers are thinking
        Angry,      // a buyer lowered their offer
        Anxious,    // a buyer walked away
    };

    enum class Action { None, JumpPreparation, Jump, Landing, Slide, Kick, Cast };
    Action CurrentAction() const { return action_; }
    size_t HeartCount() const { return hearts_.size(); }
    // Text focus disables every gameplay key, not just Space. In-flight actions still finish.
    void Update(sa::Backend& backend, float dt, bool controlsEnabled = true);
    void Draw(sa::Backend& backend) const;
    float X() const { return x_; }
    float Y() const { return y_; }
    float Lift() const { return lift_; }
    // Shows a face for `seconds`, unless a stronger one is showing (Anxious > Angry > Sheepish). Faces are
    // drawn only while she stands facing the camera, and only if their drawings exist (girl_<face>.png).
    void Feel(Face face, float seconds = 2.0f);
    void Calm() { faceLeft_ = 0.0f; }

private:
    enum class Motion
    {
        Standing,
        Starting,       // from standing into a run, sideways
        Running,
        Stopping,       // out of a run, turning back to the camera
        Recovering,     // finish an action by relaxing the arms and turning back to idle
        WalkStopping,   // finish the current half-step before planting both feet
        WalkingAway,    // up the screen, seen from behind
        WalkingToward,  // down the screen, facing the camera
        Turning,        // between seen from behind and facing the camera
    };

    void Go(Motion motion);
    void Run(int way, float dt);
    void Walk(int way, float dt);
    void StopRun(float dt);
    void Recover();
    void StopWalk();

    Motion motion_ = Motion::Standing;
    bool placed_ = false;       // put at the layout's start on the first update
    float x_ = 0.0f;            // world x of her feet
    float y_ = 0.0f;            // screen y of her feet on the ground
    float lift_ = 0.0f;         // height above the ground while jumping
    float rise_ = 0.0f;         // upward speed
    int heading_ = 1;           // sideways: +1 right, -1 left
    bool fromBehind_ = false;   // standing or walking with her back to the camera
    bool turningAway_ = false;  // while Turning: towards being seen from behind
    float time_ = 0.0f;         // in the current motion
    float turnIdlePhase_ = 0.0f;
    float runPhase_ = 0.0f;
    float runStopAdvance_ = 0.0f;
    float walkStopPhase_ = 0.0f;
    float walkStopAdvance_ = 0.0f;
    Action action_ = Action::None;
    float actionTime_ = 0.0f;
    struct HeartWave
    {
        float x, y, age;
        int direction;
    };
    std::vector<HeartWave> hearts_;
    bool castReleased_ = false;
    Face face_ = Face::None;
    float faceLeft_ = 0.0f;
};
