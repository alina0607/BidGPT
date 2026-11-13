"""LLM-written training data for buyer GPT v6 (docs/design-v2.md §6.3d).

The teacher's outputs are raw, append-only source data under ml/data/raw/teacher_gen/; labels and splits
are recomputed from them by code, so an engine change means rebuilding, not regenerating.
"""
