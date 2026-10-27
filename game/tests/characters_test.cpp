// Buyer art loading and the mood → animation rules, on the headless backend with the real assets.

#include "Characters.h"

#include <cstdio>
#include <string>

namespace
{

int failures = 0;

void Check(bool ok, const char* what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    failures += ok ? 0 : 1;
}

} // namespace

int main()
{
    auto backend = sa::CreateBackend("null");
    CharacterArt art;
    art.Load(*backend, AssetDir() + "/characters");
    Check(art.LookCount() == 8, "all 8 looks load");
    Check(art.LookFor("Maya") == art.LookFor("Maya") && art.LookFor("Maya") != nullptr,
          "a buyer always gets the same look");
    const Look& look = *art.LookFor("Maya");
    Check(look.states.size() == 6, "a look has the 6 states of docs/art.md");

    BuyerSprite sprite;
    sprite.SetMood("talk", look);
    Check(sprite.State() == "talk", "the asker talks");
    sprite.SetMood("happy", look);
    sprite.Update(0.2f, look);
    Check(sprite.State() == "happy" && sprite.Frame() == 2, "a good answer plays happy (0.2 s: frame 2 of 6)");
    sprite.Update(1.0f, look);
    Check(sprite.State() == "idle", "and settles back to idle when it is done");
    sprite.SetMood("idle", look);
    Check(sprite.State() == "idle", "the same mood again changes nothing");

    sprite.SetMood("left", look);
    Check(sprite.State() == "leave" && sprite.Visible(), "a buyer who walks away plays leave");
    sprite.Update(1.0f, look);
    Check(!sprite.Visible(), "and is gone once it is done");

    BuyerSprite winner;
    winner.SetMood("won", look);
    winner.Update(5.0f, look);
    Check(winner.State() == "win" && winner.Frame() == 5 && winner.Visible(), "the winner holds the last frame");

    CharacterArt none;
    none.Load(*backend, "/nonexistent");
    Check(none.LookCount() == 0 && none.LookFor("Maya") == nullptr, "no art: no looks, and no crash");

    std::printf(failures ? "%d failure(s)\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
