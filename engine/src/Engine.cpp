#include "sa/Engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <utility>

namespace sa
{

Engine::Engine(std::unique_ptr<Backend> backend, EngineConfig config)
    : backend_(std::move(backend)), config_(std::move(config))
{
}

Engine::~Engine() = default;

void Engine::SetScene(std::unique_ptr<Scene> scene)
{
    // Swapped at the top of the next frame so a scene can replace itself safely.
    pendingScene_ = std::move(scene);
}

bool Engine::Run()
{
    using Clock = std::chrono::steady_clock;

    if (!backend_->Init(config_.window))
    {
        std::cerr << "Failed to start backend '" << backend_->Name() << "'\n";
        return false;
    }

    running_ = true;
    auto previous = Clock::now();
    // Capture-only deterministic clock: PNG encoding must not skip animation drawings in QA GIFs.
    // Ordinary gameplay always keeps its real wall-clock delta.
    double captureDt = 0.0;
    if (std::getenv("SA_CAPTURE"))
        if (const char* value = std::getenv("SA_CAPTURE_DT"))
        {
            const double requested = std::atof(value);
            if (std::isfinite(requested) && requested >= 1.0 / 240 && requested <= config_.maxDeltaTime)
                captureDt = requested;
        }

    while (running_)
    {
        if (pendingScene_)
        {
            if (scene_)
                scene_->OnExit(*this);
            scene_ = std::move(pendingScene_);
            scene_->OnEnter(*this);
        }
        if (!scene_)
            break;

        if (!backend_->BeginFrame(config_.clearColor))
            break;

        const auto now = Clock::now();
        const double dt = captureDt > 0 ? captureDt :
            std::min(std::chrono::duration<double>(now - previous).count(), config_.maxDeltaTime);
        previous = now;

        scene_->Update(*this, dt);
        scene_->Render(*this);
        backend_->EndFrame();
    }

    if (scene_)
        scene_->OnExit(*this);
    scene_.reset();
    backend_->Shutdown();
    return true;
}

} // namespace sa
