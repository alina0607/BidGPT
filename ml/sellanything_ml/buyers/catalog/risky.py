"""Items labelled 'risky' (2026-10-04 review): dangerous animals, weapons and black-market goods. They
can be sold, but some buyers won't touch them (engine.refuses_risky); the rest negotiate as usual.
Every one has a legality question, whose shady answer is the black-market price."""

from . import attr, condition, health, item, opt, temperament


def animal_age(noun: str, young: str, adult: str, old: str) -> dict:
    return attr(f"How old is the {noun}?",
                opt("young", young, 1.3, family=1, low_maintenance=-1),
                opt("adult", adult, 1.0, performance=1),
                opt("old", old, 0.5, condition=-1))


def permits(noun: str) -> dict:
    return attr(f"Do you have the permits for the {noun}?",
                opt("permits", "Yes, every permit, all legal.", 1.3, status=1, low_maintenance=1),
                opt("none", "No papers. Cash only, no questions.", 0.5, status=-1, family=-2))


def registered(noun: str) -> dict:
    return attr(f"Is the {noun} registered?",
                opt("registered", "Registered and legal, with the paperwork.", 1.2, status=1),
                opt("unregistered", "It's not registered anywhere.", 0.6, status=-1, family=-2))


def big_animal(noun: str, low: float, typical: float, high: float, young: str, adult: str, old: str,
               gentle: str, lively: str, wild: str) -> dict:
    return item(noun, low, typical, high,
                age=animal_age(noun, young, adult, old),
                temperament=temperament(noun, gentle, lively, wild),
                health=health(noun),
                permits=permits(noun))


ITEMS = {
    "tiger": big_animal("tiger", 2000, 10000, 50000, "A six-month-old cub.", "Five years old.", "Fifteen years old.",
                        "Raised by people, gentle as a house cat.", "Playful, but rough.", "Wild. It has attacked a keeper."),
    "lion": big_animal("lion", 2000, 9000, 40000, "A cub, eight months old.", "A six-year-old male.", "An old lion, seventeen.",
                       "Hand-raised and calm.", "Lazy, but loud at night.", "Aggressive with strangers."),
    "bear": big_animal("bear", 1000, 5000, 20000, "A yearling cub.", "Seven years old.", "Twenty years old.",
                       "Circus-trained and calm.", "Curious and strong.", "Dangerous, it charges."),
    "crocodile": big_animal("crocodile", 300, 2000, 10000, "A two-foot hatchling.", "Eight feet long.", "An old twelve-footer.",
                            "Sluggish, mostly sleeps.", "Snaps at food.", "It has bitten someone."),
    "elephant": big_animal("elephant", 10000, 40000, 150000, "A young calf.", "Twenty years old.", "Fifty years old.",
                           "Gentle, she loves people.", "Playful and very strong.", "Unpredictable, stay back."),
    "rhinoceros": big_animal("rhinoceros", 15000, 60000, 250000, "A calf.", "A twelve-year-old male.", "Forty years old.",
                             "Docile around keepers.", "Grumpy in the mornings.", "It charges anything that moves."),
    "giraffe": big_animal("giraffe", 10000, 40000, 100000, "A calf, six feet tall.", "Fourteen feet tall, adult.", "Twenty-two years old.",
                          "Gentle, eats from your hand.", "Skittish.", "Kicks hard when scared."),
    "zebra": big_animal("zebra", 2000, 8000, 20000, "A foal.", "Six years old.", "Twenty years old.",
                        "Halter trained.", "Nervous around people.", "It bites and kicks."),
    "kangaroo": big_animal("kangaroo", 1000, 3000, 8000, "A joey, still bottle fed.", "Four years old.", "Fifteen years old.",
                           "Tame, follows you around.", "Bouncy and curious.", "A boxer. It has kicked someone."),
    "monkey": item(
        "pet monkey", 2000, 7000, 25000,
        species=attr("What kind of monkey is it?",
                     opt("capuchin", "A capuchin.", 1.0, family=1),
                     opt("marmoset", "A tiny marmoset.", 0.8, space=2),
                     opt("macaque", "A macaque.", 0.7, family=-2, low_maintenance=-1)),
        age=animal_age("monkey", "A baby, still in diapers.", "Five years old.", "Twenty years old."),
        temperament=temperament("monkey", "Sweet and cuddly.", "Mischievous, opens everything.", "It bites when upset."),
        permits=permits("monkey"),
    ),
    "scorpion": item(
        "pet scorpion", 20, 60, 300,
        species=attr("What kind of scorpion is it?",
                     opt("emperor", "An emperor scorpion, mild venom.", 1.0, family=1, low_maintenance=1),
                     opt("deathstalker", "A deathstalker. Very venomous.", 2.0, status=2, family=-2)),
        temperament=temperament("scorpion", "Calm, rarely stings.", "Twitchy.", "Defensive, stings fast."),
        enclosure=attr("Does the scorpion come with an enclosure?",
                       opt("yes", "Yes, with a heat mat.", 1.3, low_maintenance=1),
                       opt("no", "Just the scorpion in a box.", 0.8, low_maintenance=-1)),
    ),
    "rifle": item(
        "hunting rifle", 200, 700, 3000,
        type=attr("What kind of rifle is it?",
                  opt("bolt", "A bolt-action deer rifle.", 1.0, active=2, performance=1),
                  opt("antique", "An antique lever-action from 1890.", 2.5, status=2, style=2),
                  opt("22", "A small .22 for target practice.", 0.6, performance=-1)),
        condition=condition("rifle", broken="The bolt sticks."),
        scope=attr("Does the rifle come with a scope?",
                   opt("scope", "Yes, with a good scope.", 1.3, performance=2),
                   opt("none", "Iron sights only.", 0.9, style=1)),
        registered=registered("rifle"),
    ),
    "cannon": item(
        "antique cannon", 500, 3000, 30000,
        era=attr("How old is the cannon?",
                 opt("civil", "A Civil War field cannon.", 2.5, status=2, style=1),
                 opt("replica", "A newer replica for reenactments.", 0.6, performance=1),
                 opt("signal", "A small brass signal cannon from a yacht.", 0.8, space=2, style=2)),
        fires=attr("Does the cannon still fire?",
                   opt("yes", "Yes, black powder, very loud.", 1.2, performance=2, family=-2),
                   opt("no", "No, it's decorative now.", 0.9, family=1)),
        condition=condition("cannon", broken="The carriage wheels are rotten."),
    ),
    "sword": item(
        "sword", 50, 300, 5000,
        type=attr("What kind of sword is it?",
                  opt("katana", "A hand-forged katana.", 2.0, style=2, status=1),
                  opt("antique", "A 19th-century cavalry saber.", 3.0, status=2),
                  opt("replica", "A movie replica, for display.", 0.4, style=1)),
        sharp=attr("Is the sword sharp?",
                   opt("sharp", "Razor sharp.", 1.2, performance=2, family=-2),
                   opt("blunt", "Blunt, display only.", 0.9, family=1)),
        condition=condition("sword", broken="The blade has a nick and some rust."),
    ),
    "passport": item(
        "passport", 500, 2000, 10000,
        validity=attr("Is the passport still valid?",
                      opt("valid", "Valid for another six years.", 1.5, performance=2),
                      opt("expired", "It expired last year.", 0.3, performance=-2)),
        source=attr("Whose passport is it?",
                    opt("blank", "It's blank. Don't ask where it's from.", 1.4, status=-2, family=-2),
                    opt("old", "My own old one, cancelled. A souvenir.", 0.05, style=1, status=1)),
        country=attr("Which country is the passport from?",
                     opt("strong", "One that gets you visa-free almost anywhere.", 1.6, performance=2),
                     opt("weak", "A small country nobody checks.", 0.7, performance=-1)),
    ),
    "skull": item(
        "skull", 50, 400, 5000,
        kind=attr("What kind of skull is it?",
                  opt("human", "A real human skull. Old medical school stock, supposedly.", 3.0, status=2, family=-2),
                  opt("animal", "A bison skull from a ranch.", 1.0, style=2),
                  opt("replica", "A resin replica.", 0.2, family=1)),
        condition=condition("skull", broken="The jaw is missing."),
        papers=attr("Is there any paperwork for the skull?",
                    opt("papers", "Yes, the provenance papers.", 1.4, status=1),
                    opt("none", "No papers at all.", 0.6, status=-1)),
    ),
}
