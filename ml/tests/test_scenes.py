"""Buyer scenes (docs/design-v2.md §5.4): props.json, the scoring rule, the layout and the protocol."""
from types import SimpleNamespace

import pytest

from sellanything_ml.buyers import scenes
from sellanything_ml.buyers.schema import load_items, load_traits
from sellanything_ml.server import Models, handle_message
from sellanything_ml.sketch.data import load_categories

TAGS = {"condition", "space", "low_maintenance", "family", "status", "performance", "style", "active"}


@pytest.fixture(scope="module")
def world():
    props, groups = scenes.tables()
    return SimpleNamespace(items=load_items(), traits=load_traits(), props=props, groups=groups)


def buyer(world, *traits):
    weights: dict[str, int] = {}
    for t in traits:
        for tag, w in world.traits.traits[t].weights.items():
            weights[tag] = min(2, weights.get(tag, 0) + w)
    return SimpleNamespace(traits=list(traits), weights=weights)


def props_of(world, item, *traits, answered=None):
    objs = scenes.plan(world.items[item], world.groups[item], buyer(world, *traits), world.traits,
                       world.props, answered=answered)
    return objs[0], objs[1:]


def test_props_json_names_tags_zones_sizes_and_groups_are_all_known(world):
    categories = set(load_categories())
    groups = set(world.groups.values())
    for name, p in world.props.items():
        assert name in categories, name
        assert set(p.tags) <= TAGS and all(-2 <= v <= 2 for v in p.tags.values()), name
        assert p.zone in scenes.ZONES and p.size in scenes.SIZES, name
        assert p.fits == "*" or set(p.fits) <= groups, name


def test_zone_boxes_are_inside_the_canvas_and_never_overlap():
    boxes = [scenes.ITEM_BOX, *scenes.ZONES.values()]
    for x, y, w, h in boxes:
        assert x >= 0 and y >= 0 and x + w <= scenes.SCENE_W and y + h <= scenes.SCENE_H
    for i, (ax, ay, aw, ah) in enumerate(boxes):
        for bx, by, bw, bh in boxes[i + 1:]:
            assert ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay
    assert max(w * h for _, _, w, h in boxes) == scenes.ITEM_BOX[2] * scenes.ITEM_BOX[3]   # the item is largest


def test_an_outdoorsy_buyer_pictures_a_dog_outdoors_but_not_a_cat(world):
    item, props = props_of(world, "dog", "outdoorsy")
    assert item["label"] == "dog" and item["role"] == "item"
    outdoor = [p for p in props if world.props[p["label"]].tags.get("active", 0) > 0]
    assert outdoor and all(p["reason"].startswith("Outdoorsy") for p in outdoor)
    _, cat_props = props_of(world, "cat", "outdoorsy")            # a cat's profile has no `active`
    assert not any(world.props[p["label"]].tags.get("active", 0) > 0 for p in cat_props)


def test_answers_change_the_scene_a_calm_senior_dog_loses_the_outdoors(world):
    _, props = props_of(world, "dog", "outdoorsy", answered={"age": "senior", "temperament": "calm"})
    assert not any(world.props[p["label"]].tags.get("active", 0) > 0 for p in props)


def test_pet_food_shows_up_whoever_the_buyer_is_and_never_for_other_items(world):
    for traits in [("apartment_dweller",), ("collector",), ("retiree",)]:
        assert "steak" in [p["label"] for p in props_of(world, "dog", *traits)[1]], traits
    assert "steak" not in [p["label"] for p in props_of(world, "bicycle", "retiree")[1]]


def test_at_most_three_props_one_per_zone_positive_scores_and_never_the_item_itself(world):
    for item in ["dog", "cat", "bicycle", "laptop", "guitar", "couch"]:
        for trait in world.traits.traits:
            _, props = props_of(world, item, trait)
            assert len(props) <= scenes.MAX_PROPS
            assert len({p["zone"] for p in props}) == len(props)
            assert all(p["score"] > 0 and p["label"] != item for p in props)


def test_fit_scales_uniformly_and_stays_in_the_box():
    strokes = [([0, 100, 100], [0, 0, 50])]                 # 100 wide, 50 tall
    for anchor in ("center", "bottom"):
        out = scenes.fit(strokes, (10, 20, 60, 90), 0.8, anchor)
        xs, ys = out[0]
        assert min(xs) >= 10 and max(xs) <= 70 and min(ys) >= 20 and max(ys) <= 110
        assert (max(xs) - min(xs)) / (max(ys) - min(ys)) == pytest.approx(2.0, rel=0.02)   # never stretched
    assert scenes.fit(strokes, (10, 20, 60, 90), 1.0, "bottom")[0][1][-1] == pytest.approx(110)


def test_scene_protocol_draws_every_object_inside_its_box():
    models = Models(sketch_label="dog")                      # test drawer: the same small drawing
    state = handle_message({"type": "round_start", "session": "s", "item": "dog", "buyers": 3, "seed": 1,
                            "voice": "phrases"}, models)["state"]
    reply = handle_message({"type": "scene", "session": "s", "buyer": state["buyers"][0]["name"], "seed": 2}, models)
    assert reply["type"] == "scene_result" and reply["ok"]
    assert (reply["width"], reply["height"]) == (scenes.SCENE_W, scenes.SCENE_H)
    assert reply["objects"][0]["label"] == "dog"
    for o in reply["objects"]:
        x, y, w, h = o["box"]
        for xs, ys in o["strokes"]:
            assert all(x - 0.01 <= v <= x + w + 0.01 for v in xs) and all(y - 0.01 <= v <= y + h + 0.01 for v in ys)


def test_scene_errors_unknown_session_buyer_or_a_refused_item():
    models = Models(sketch_label="dog")
    assert handle_message({"type": "scene", "session": "none", "buyer": "A"}, models)["type"] == "error"
    handle_message({"type": "round_start", "session": "s", "item": "dog", "voice": "phrases"}, models)
    assert "no buyer named Nobody" in handle_message({"type": "scene", "session": "s", "buyer": "Nobody"}, models)["message"]
    assert handle_message({"type": "scene", "session": "s"}, models)["type"] == "error"
    handle_message({"type": "round_start", "session": "r", "item": "aircraft carrier", "voice": "phrases"}, models)
    name = handle_message({"type": "round_start", "session": "r", "item": "aircraft carrier",
                           "voice": "phrases"}, models)["state"]["buyers"][0]["name"]
    assert "won't buy" in handle_message({"type": "scene", "session": "r", "buyer": name}, models)["message"]
