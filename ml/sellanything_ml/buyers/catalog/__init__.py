"""Source of ml/data/item_schema.json for the items not written by hand.

Each module in this package defines ITEMS: {quickdraw_class: item dict}. Shared attribute builders
below keep wording and numbers consistent across items; every item still gets at least one attribute
specific to it. Regenerate the JSON with:

    python -m sellanything_ml.buyers.catalog.build
"""

from __future__ import annotations


def opt(id: str, answer: str, factor: float = 1.0, **signals: int) -> dict:
    return {"id": id, "answer": answer, "factor": factor, "signals": signals}


def attr(question: str, *options: dict) -> dict:
    return {"question": question, "options": list(options)}


def item(noun: str, low: float, typical: float, high: float, **attributes: dict) -> dict:
    return {"noun": noun, "price": {"low": low, "typical": typical, "high": high},
            "attributes": attributes}


# --- shared attributes ----------------------------------------------------------------------

def condition(noun: str, *, broken: str | None = None) -> dict:
    """Generic used-goods condition."""
    return attr(
        f"What condition is the {noun} in?",
        opt("like_new", "Like new, barely used.", 1.2, condition=2),
        opt("used", "Used, but in good shape.", 1.0),
        opt("worn", "Pretty worn, it shows its age.", 0.7, condition=-1),
        opt("damaged", broken or "It's damaged, honestly.", 0.35, condition=-2),
    )


def works(noun: str, *, faulty: str, dead: str) -> dict:
    """Does a device/appliance work?"""
    return attr(
        f"Does the {noun} work properly?",
        opt("perfect", "Works perfectly.", 1.1, performance=2),
        opt("faulty", faulty, 0.6, performance=-1, low_maintenance=-1),
        opt("dead", dead, 0.2, performance=-2, condition=-1),
    )


def brand(noun: str, known: str, *, premium_factor: float = 1.5, generic: str = "A no-name brand.") -> dict:
    return attr(
        f"What brand is the {noun}?",
        opt("known", f"{known}.", premium_factor, status=2, performance=1),
        opt("generic", generic, 0.75, status=-1),
        opt("unknown", "Not sure, there's no label.", 0.9),
    )


def age(noun: str, *, new: str = "Less than a year old.", mid: str = "About three years old.",
        old: str = "Over ten years old.", vintage: bool = False) -> dict:
    return attr(
        f"How old is the {noun}?",
        opt("new", new, 1.2, condition=1),
        opt("mid", mid, 1.0),
        opt("old", old, 1.3 if vintage else 0.6, **({"status": 2, "style": 1} if vintage else {"condition": -1})),
    )


def size(noun: str, small: str, medium: str, large: str, *, large_factor: float = 1.2) -> dict:
    return attr(
        f"How big is the {noun}?",
        opt("small", small, 0.8, space=2),
        opt("medium", medium, 1.0),
        opt("large", large, large_factor, space=-2, performance=1),
    )


def freshness(noun: str, *, stale: str, spoiled: str) -> dict:
    """Food freshness."""
    return attr(
        f"How fresh is the {noun}?",
        opt("today", "Made fresh today.", 1.3, condition=2),
        opt("day_old", "From yesterday, still good.", 0.9, condition=-1),
        opt("stale", stale, 0.4, condition=-2),
        opt("spoiled", spoiled, 0.1, condition=-2, family=-2),
    )


def produce_freshness(noun: str) -> dict:
    """Fruit and vegetables."""
    return attr(
        f"How fresh is the {noun}?",
        opt("just_picked", "Picked this morning.", 1.3, condition=2),
        opt("few_days", "A few days old, still crisp.", 1.0, condition=1),
        opt("week", "About a week old, getting soft.", 0.6, condition=-1),
        opt("going_bad", "Starting to go bad, honestly.", 0.25, condition=-2),
    )


def organic(noun: str) -> dict:
    return attr(
        f"Is the {noun} organic?",
        opt("yes", "Yes, certified organic.", 1.3, family=1, status=1),
        opt("no", "No, regular.", 1.0),
    )


def quantity(question: str, few: str, normal: str, lots: str) -> dict:
    return attr(
        question,
        opt("few", few, 0.5, performance=-1),
        opt("normal", normal, 1.0),
        opt("lots", lots, 1.8, family=1, performance=1),
    )


def extras(noun: str, included: str) -> dict:
    return attr(
        f"Does the {noun} come with anything else?",
        opt("extras", included, 1.2, low_maintenance=1, performance=1),
        opt("nothing", f"Just the {noun} itself.", 1.0),
    )


def material(noun: str, fine: str, mid: str, cheap: str) -> dict:
    return attr(
        f"What's the {noun} made of?",
        opt("fine", fine, 1.4, style=2, condition=1),
        opt("mid", mid, 1.0, low_maintenance=1),
        opt("cheap", cheap, 0.6, style=-1),
    )


def new_only(noun: str) -> dict:
    """Hygiene items: only unopened ones have real value."""
    return attr(
        f"Is the {noun} new?",
        opt("sealed", "Brand new, still sealed.", 1.0, condition=2, family=1),
        opt("opened", "Opened, but never used.", 0.5, condition=-1),
        opt("used", "It's been used.", 0.05, condition=-2, family=-2),
    )


def health(noun: str) -> dict:
    """Animals."""
    return attr(
        f"Is the {noun} healthy? Seen a vet recently?",
        opt("checked", "Healthy, vet checked last month.", 1.2, condition=2, family=1),
        opt("unknown", "Seems fine, never been to a vet.", 0.85, condition=-1),
        opt("sick", "It's been a bit sick lately.", 0.4, condition=-2, low_maintenance=-2),
    )


def temperament(noun: str, gentle: str, lively: str, wild: str) -> dict:
    return attr(
        f"What's the {noun}'s temperament like?",
        opt("gentle", gentle, 1.15, family=2, low_maintenance=1),
        opt("lively", lively, 1.0, active=2, low_maintenance=-1),
        opt("wild", wild, 0.7, family=-2),
    )


def power_source(noun: str) -> dict:
    return attr(
        f"Is the {noun} corded or battery-powered?",
        opt("cordless", "Cordless, comes with two batteries.", 1.3, low_maintenance=1, active=1),
        opt("corded", "Corded.", 0.9, performance=1),
    )
