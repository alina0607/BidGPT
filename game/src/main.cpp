#include "GameContext.h"
#include "Scenes.h"

#include "sa/Engine.h"

#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace
{

void PrintUsage()
{
    std::cout << "Usage: sell_anything [--backend NAME] [--server HOST:PORT] [--autoplay] [--list-backends]\n"
                 "Backends in this build:";
    for (std::string_view name : sa::AvailableBackends())
        std::cout << ' ' << name;
    std::cout << " (default: " << sa::AvailableBackends().front() << ")\n"
              << "The ML server must be running: cd ml && python3 -m sellanything_ml.server\n";
}

struct Options
{
    std::string_view backend;
    std::string host = "127.0.0.1";
    int port = 5555;
    bool autoplay = false;
};

// "host" or "host:port"; false if the port is not a number in 1..65535.
bool ParseServer(const std::string& value, Options& options)
{
    const size_t colon = value.rfind(':');
    options.host = value.substr(0, colon);
    if (colon == std::string::npos)
        return true;
    const std::string digits = value.substr(colon + 1);
    if (digits.empty() || digits.size() > 5 || digits.find_first_not_of("0123456789") != std::string::npos)
        return false;
    options.port = std::stoi(digits);
    return options.port >= 1 && options.port <= 65535;
}

// Fills `options`; returns the exit code to stop with (help, or a bad argument), if any.
std::optional<int> ParseArgs(int argc, char** argv, Options& options)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg = argv[i];
        const bool hasValue = i + 1 < argc;
        if (arg == "--backend" && hasValue)
            options.backend = argv[++i];
        else if (arg == "--server" && hasValue)
        {
            const std::string value = argv[++i];
            if (!ParseServer(value, options))
            {
                std::cerr << "Bad port in --server " << value << '\n';
                return 2;
            }
        }
        else if (arg == "--autoplay")
            options.autoplay = true;
        else if (arg == "--list-backends" || arg == "--help")
        {
            PrintUsage();
            return 0;
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << '\n';
            PrintUsage();
            return 2;
        }
    }
    return std::nullopt;
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    if (const auto exitCode = ParseArgs(argc, argv, options))
        return *exitCode;
    auto backend = sa::CreateBackend(options.backend);
    if (!backend)
    {
        std::cerr << "Backend '" << options.backend << "' is not available in this build\n";
        PrintUsage();
        return 2;
    }
    std::cout << "Using backend: " << backend->Name() << ", ML server " << options.host << ':' << options.port << '\n';

    GameContext ctx(options.host, options.port);
    ctx.autoplay = options.autoplay;
    sa::EngineConfig config;
    config.window.width = static_cast<int>(kWidth);
    config.window.height = static_cast<int>(kHeight);
    sa::Engine engine(std::move(backend), config);
    engine.SetScene(MakeCanvasScene(ctx));
    const bool ran = engine.Run();
    if (!options.autoplay)
        return ran ? 0 : 1;
    // Autoplay is a test: it passes only if it got all the way to a sale.
    std::cout << (ctx.sold ? "autoplay: sold\n" : "autoplay: ended without a sale\n");
    return ran && ctx.sold ? 0 : 3;
}
