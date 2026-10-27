"""Buyer lines from the trained model, with the engine as a guardrail, and (models trained on the
`parse` task) which option a typed answer means.

The model writes what buyers say. The engine still owns the numbers: a reaction is only used if
the model's own interest change agrees in sign with the engine's, so words never contradict the
offer on screen. Anything malformed falls back to the v1 phrase pools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import format as fmt
from . import phrases

ML_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINTS = [ML_DIR / "checkpoints" / "buyer_gpt.pt", ML_DIR / "runs" / "sft" / "best.pt"]


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


@dataclass
class VoiceStats:
    used: int = 0
    fallback: dict[str, int] = field(default_factory=dict)

    def miss(self, reason: str) -> None:
        self.fallback[reason] = self.fallback.get(reason, 0) + 1


class ModelVoice:
    prompt_format = 1        # the checkpoint's; format 2 puts the engine's question in the ask prompt

    def __init__(self, checkpoint: Path, device: str = "cpu", max_new_tokens: int = 60):
        from ..llm.evaluate import load_model   # torch only when a model is actually used
        import torch

        self.torch = torch
        self.model, self.tok = load_model(checkpoint, torch.device(device))
        self.prompt_format = self.model.prompt_format
        # reads typed answers only if trained on parse AND its evaluation beat the rule matcher (llm/export.py)
        self.understands = "parse" in self.model.tasks and self.model.parse_ready
        self.max_new = max_new_tokens
        self.stats = VoiceStats()
        self.checkpoint = checkpoint

    def _complete(self, prompt: str, temperature: float = 0.7) -> str:
        ids = self.torch.tensor([self.tok.encode(prompt)])
        out = self.model.generate(ids, self.max_new, temperature=temperature, top_k=40,
                                  stop_id=self.tok.eos_id)
        return self.tok.decode(out[0, ids.size(1):].tolist())

    def ask(self, view: fmt.BuyerView, name: str, attribute: str, question: str) -> str | None:
        given = None
        if self.prompt_format >= 3:
            given = fmt.to_slot(question, list(view.names) or fmt.item_names(view.noun, view.noun))
        elif self.prompt_format == 2:
            given = question
        out = fmt.parse_line(self._complete(fmt.ask_prompt(view, attribute, given)))
        if out is None or out[0] != name:
            self.stats.miss("ask_format")
            return None
        line = fmt.fill_slot(out[1], view.noun)
        # The engine decided what to ask; the line must still ask it.
        if not fmt.on_topic(line, question, view.noun):
            self.stats.miss("ask_off_topic")
            return None
        # "My place is pretty small" only from someone whose traits make space matter.
        if view.cares and not phrases.ask_reasons_in(line) <= set(view.cares):
            self.stats.miss("ask_wrong_reason")
            return None
        self.stats.used += 1
        return line

    def react(self, view: fmt.BuyerView, name: str, engine_delta: int,
              reasons: set[tuple[str, int]] | None = None) -> str | None:
        """`reasons`: the (tag, +1/-1) this answer really means to this buyer; a reason outside it
        ("too big" about a kitten's age) rejects the line. None skips the check."""
        out = fmt.parse_react(self._complete(fmt.react_prompt(view)))
        if out is None or out[1] is None or out[0] != name:
            self.stats.miss("react_format")
            return None
        if _sign(out[2]) != _sign(engine_delta):
            self.stats.miss("react_disagrees")
            return None
        if reasons is not None and not phrases.react_reasons_in(out[1]) <= reasons:
            self.stats.miss("react_wrong_reason")
            return None
        self.stats.used += 1
        return fmt.fill_slot(out[1], view.noun)

    def parse(self, noun: str, attr, typed: str) -> str | None:
        """The option id the player meant, or None when the model reads none of them (the buyer then
        asks again). Greedy: the same answer always means the same thing."""
        prompt = fmt.parse_prompt(noun, attr.question, [o.answer for o in attr.options], typed)
        got = fmt.read_parse(self._complete(prompt, temperature=0), len(attr.options))
        if got is False:
            self.stats.miss("parse_format")
            return None
        self.stats.used += 1
        return None if got is None else attr.options[got].id

    def refuse(self, noun: str, persona: str, name: str) -> str | None:
        out = fmt.parse_line(self._complete(fmt.refuse_prompt(noun, persona)))
        if out is None or out[0] != name:
            self.stats.miss("refuse_format")
            return None
        self.stats.used += 1
        return fmt.fill_slot(out[1], noun)


def load_default(device: str = "cpu") -> ModelVoice | None:
    for path in DEFAULT_CHECKPOINTS:
        if path.exists():
            return ModelVoice(path, device)
    return None
