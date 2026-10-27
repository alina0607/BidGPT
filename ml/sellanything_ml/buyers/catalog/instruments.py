from . import age, attr, brand, condition, extras, item, opt


def level(noun: str) -> dict:
    return attr(f"Is the {noun} a student or professional model?",
                opt("student", "A student model.", 0.6, family=1),
                opt("intermediate", "Intermediate.", 1.0, performance=1),
                opt("pro", "A professional model.", 3.0, performance=2, status=2))


def playable(noun: str, issue: str, broken: str) -> dict:
    return attr(f"Does the {noun} play well?",
                opt("great", "Plays beautifully, just serviced.", 1.2, performance=2, condition=1),
                opt("issue", issue, 0.75, performance=-1, low_maintenance=-1),
                opt("broken", broken, 0.3, performance=-2, condition=-2))


ITEMS = {
    "cello": item(
        "cello", 150, 1200, 15000,
        level=level("cello"),
        size=attr("What size is it?",
                  opt("full", "Full size.", 1.0, performance=1),
                  opt("half", "Half size, for kids.", 0.5, family=2, space=1)),
        playable=playable("cello", "There's a buzz on the C string.", "There's a crack in the top."),
        extras=extras("cello", "A hard case and a carbon fiber bow."),
    ),
    "clarinet": item(
        "clarinet", 40, 300, 4000,
        level=level("clarinet"),
        material=attr("Is it wood or plastic?",
                      opt("wood", "Grenadilla wood.", 1.5, style=1, status=1, low_maintenance=-1),
                      opt("plastic", "Plastic, very durable.", 0.8, low_maintenance=2, family=1)),
        brand=brand("clarinet", "Buffet"),
        playable=playable("clarinet", "A pad needs replacing.", "Several keys are bent."),
    ),
    "drums": item(
        "drum kit", 50, 500, 5000,
        type=attr("What kind of drum kit is it?",
                  opt("acoustic", "A five-piece acoustic kit.", 1.0, performance=1, space=-2, family=-1),
                  opt("electronic", "An electronic kit with headphones.", 1.2, space=1, family=2),
                  opt("kids", "A kids' starter kit.", 0.3, family=2)),
        cymbals=attr("Does it come with cymbals?",
                     opt("pro", "Pro cymbals and hardware.", 1.8, performance=2, status=1),
                     opt("basic", "Basic cymbals.", 1.0),
                     opt("none", "No cymbals.", 0.7)),
        brand=brand("drums", "Pearl"),
        condition=condition("drum kit", broken="The snare head is torn."),
    ),
    "harp": item(
        "harp", 300, 3000, 40000,
        type=attr("What kind of harp is it?",
                  opt("lap", "A small lap harp.", 0.3, space=2),
                  opt("lever", "A thirty-six string lever harp.", 1.0, performance=1),
                  opt("pedal", "A full concert pedal harp.", 8.0, status=2, space=-2)),
        playable=playable("harp", "A few strings need replacing.", "The soundboard is bulging."),
        extras=extras("harp", "A padded cover and a tuning key."),
    ),
    "piano": item(
        "piano", 100, 1500, 50000,
        type=attr("What kind of piano is it?",
                  opt("upright", "An upright piano.", 1.0, family=1, space=-1),
                  opt("grand", "A baby grand.", 5.0, status=2, style=2, space=-2),
                  opt("digital", "A digital piano with weighted keys.", 0.6, space=1, low_maintenance=2)),
        tuned=attr("When was it last tuned?",
                   opt("recent", "Tuned last month.", 1.2, performance=2),
                   opt("years", "Not for a few years.", 0.8, low_maintenance=-1),
                   opt("never", "I don't think it's ever been tuned.", 0.5, performance=-2)),
        brand=brand("piano", "Steinway", premium_factor=6.0),
        age=age("piano", old="It's from the 1920s.", vintage=True),
    ),
    "saxophone": item(
        "saxophone", 100, 800, 8000,
        type=attr("What kind of sax is it?",
                  opt("alto", "An alto sax.", 1.0, family=1),
                  opt("tenor", "A tenor sax.", 1.3, performance=1),
                  opt("soprano", "A soprano sax.", 1.2, style=1, space=1)),
        level=level("saxophone"),
        playable=playable("saxophone", "It needs a few new pads.", "The neck is bent."),
        extras=extras("saxophone", "A case, a strap and a box of reeds."),
    ),
    "trombone": item(
        "trombone", 60, 400, 4000,
        level=level("trombone"),
        slide=attr("How's the slide?",
                   opt("smooth", "Glides like butter.", 1.2, performance=2),
                   opt("sticky", "It sticks a bit.", 0.7, performance=-1),
                   opt("dented", "There's a dent in the slide.", 0.4, performance=-2, condition=-2)),
        brand=brand("trombone", "Bach"),
    ),
    "trumpet": item(
        "trumpet", 50, 400, 4000,
        level=level("trumpet"),
        valves=attr("How are the valves?",
                    opt("smooth", "Fast and smooth.", 1.2, performance=2),
                    opt("sticky", "A bit sticky.", 0.7, performance=-1, low_maintenance=-1)),
        brand=brand("trumpet", "Bach"),
        condition=condition("trumpet", broken="The bell has a dent."),
    ),
    "violin": item(
        "violin", 50, 500, 20000,
        level=level("violin"),
        size=attr("What size is it?",
                  opt("full", "Full size, four-four.", 1.0, performance=1),
                  opt("kids", "A half-size for kids.", 0.5, family=2, space=1)),
        age=age("violin", old="It's over a hundred years old.", vintage=True),
        playable=playable("violin", "The bridge needs adjusting.", "There's a crack near the f-hole."),
    ),
}
