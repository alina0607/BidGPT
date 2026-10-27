# parse_v1: how players answer a buyer's question

You are writing training data for a small language model inside a game. In the game, the player is
selling an item; a buyer asks a question; the player types an answer in their own words. The model
must learn which of the listed options the player meant.

For each request below you get the item, the question, **all** options of that attribute (so you
know what the answer must NOT be confused with) and the **target option**. Write `{n}` different
answers a real player might type that clearly mean the target option and none of the others.

Mix these styles across the `{n}` answers (label each one):

- `plain`: a normal full sentence.
- `casual`: short, lowercase, chatty, the way people type in a game ("yeah shes had all her shots").
- `specific`: a concrete instance instead of the category, using real-world knowledge (a breed name
  for "purebred", a brand for "known brand", "86%" for a battery bucket, "she's 10 weeks" for a
  kitten). Use different instances across answers.
- `indirect`: says it without the option's key words ("no papers, I got her from a shelter").
- `extra`: answers and adds a harmless detail the game does not track ("Two years old, she loves the
  window seat.").

Rules:

- English only. Do not copy the option's wording; do not reuse the same key word in every answer.
- Each answer must be unambiguous: a careful reader who sees all options picks the target.
- Digits are fine in player answers (players type them).
- Do not add typos on purpose; typos are added later by code.

Write one JSON object per request, on one line, exactly in this shape:

{"request_id": "...", "outputs": [{"text": "...", "style": "plain"}, ...]}
