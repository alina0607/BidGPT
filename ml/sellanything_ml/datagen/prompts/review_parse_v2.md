# review_parse_v2: blind check of player answers (letters only)

You are checking training data for a game. Each line gives an item, a buyer's question, the possible
answers lettered A, B, C… (in random order) and what a player typed. Nobody tells you which answer the
player meant: decide it yourself from the meaning of what they typed.

Read every line yourself. Do not write code or use keyword rules to decide: the point of this check is
an independent human-like reading, and a script would only repeat what the writer already did.

For each line write one tab-separated line:

    <id>	<letter, or ambiguous, or none>	<high or low>	<reason, at most 10 words>

- `ambiguous`: two or more answers fit. `none`: no answer fits.
- `low`: a real buyer could reasonably read it differently.
- The reason must be about this line's text (e.g. "Ragdoll is a pedigree breed"), not a template.
