from sellanything_ml.buyers.cat_dialogues import attribute_for, convert_sale


def _sale():
    return {
        "seed": 7,
        "sale": {"cat": {"months": 3, "breed": "a mixed-breed tabby", "health": "no vaccines yet",
                           "temperament": "very cuddly"}},
        "cards": {"Maya": {"persona": "lives in a small apartment", "cares_about": ["quiet"]}},
        "events": [
            {"kind": "ask", "name": "Maya", "says": "How old is the cat?"},
            {"kind": "answer", "to": "Maya", "says": "She's only three months old."},
            {"kind": "react", "name": "Maya", "old": 50, "offer": 60, "sign": "+",
             "says": "A kitten would fit right in."},
        ],
    }


def test_maps_cat_questions_to_game_attributes():
    assert attribute_for("How old is the cat?") == "age"
    assert attribute_for("Has she had her vaccinations and a vet check?") == "health"
    assert attribute_for("What colour is her favorite blanket?") is None


def test_converts_teacher_turns():
    rows = convert_sale(_sale())
    assert [row["task"] for row in rows] == ["ask", "parse", "react"]
    assert rows[1]["target"] == "A<|end|>"
    assert rows[2]["target"].endswith("<|interest|>+1<|end|>")


def test_drops_a_reaction_whose_sign_and_offer_move_in_opposite_directions():
    sale = _sale()
    sale["events"][2]["offer"] = 40
    assert [row["task"] for row in convert_sale(sale)] == ["ask", "parse"]
