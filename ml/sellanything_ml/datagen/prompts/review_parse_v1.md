# review_parse_v1: blind check of player answers

You are checking training data for a game. For each line you get an item, a buyer's question, the
attribute's options (id and canonical answer) and a player's typed answer. You are **not** told which
option the writer intended. Decide which option the player's answer means.

Answer with one JSON object per line, exactly in this shape:

{"answer_id": "...", "maps_to": "<option id>", "confidence": "high|low", "reason": "<= 15 words"}

- `maps_to` is one option id, or `"ambiguous"` if two or more options fit, or `"none"` if no option fits.
- Use `low` confidence when a real buyer could reasonably read it differently.
- Judge only the meaning, not the style.
