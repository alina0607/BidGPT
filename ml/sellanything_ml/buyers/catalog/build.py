"""Regenerate ml/data/item_schema.json from handwritten.json + every catalog module.

    python -m sellanything_ml.buyers.catalog.build
"""

from __future__ import annotations

import importlib
import json
import pkgutil
from pathlib import Path

from .. import catalog
from ..schema import DATA_DIR, SELLABLE, load_items, load_labels, load_traits, validate

HERE = Path(__file__).resolve().parent
OUT = DATA_DIR / "item_schema.json"


def collect() -> dict[str, dict]:
    items = dict(json.loads((HERE / "handwritten.json").read_text(encoding="utf-8"))["items"])
    for info in pkgutil.iter_modules(catalog.__path__):
        if info.name == "build":
            continue
        module = importlib.import_module(f"{catalog.__name__}.{info.name}")
        for key, spec in module.ITEMS.items():
            if key in items:
                raise SystemExit(f"{key!r} defined twice (second time in {info.name})")
            items[key] = spec
    return dict(sorted(items.items()))


def main() -> None:
    items = collect()
    OUT.write_text(json.dumps({"version": 1, "currency": "USD", "items": items}, indent=1,
                              ensure_ascii=False) + "\n", encoding="utf-8")
    errors = validate(load_items(OUT), load_traits(), load_labels())
    labels = load_labels()
    missing = sorted(k for k, v in labels.items() if v in SELLABLE and k not in items)
    print(f"{len(items)} items written to {OUT.relative_to(DATA_DIR.parent)}; "
          f"{len(missing)} sellable items still without a schema")
    for e in errors:
        print("ERROR", e)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
