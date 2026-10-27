"""Architecture diagrams of the two transformers, in English and Chinese, drawn from their configs.

    cd ml && python tools/draw_architecture.py            # → docs/architecture/{buyer_gpt,brush_gpt}_{en,zh}.png

Buyer GPT = runs/sft_b2/best.pt (llm/model.py, prompt format 5); brush GPT = runs/brushgpt/I_aligned_points_eyes
(sketch/brushgpt.py). The numbers on the boxes are read from the checkpoints' configs, not typed in.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))
OUT = ML.parent / "docs" / "architecture"
COLOR = {"io": "#d9daf5", "embed": "#eaa79b", "pos": "#f8d9a6", "attn": "#cfe3cf", "norm": "#fff8d2",
         "ffn": "#cde5f7", "out": "#eaa79b", "note": "#f4f4f4", "data": "#f7d3e6"}
FONT = {"en": ["Helvetica", "Arial Unicode MS", "DejaVu Sans"], "zh": ["Heiti TC", "Arial Unicode MS"]}


class Canvas:
    """A figure in box units: boxes, the plus circles, arrows and dotted containers like a paper's diagram."""

    def __init__(self, width: float, height: float, lang: str):
        self.fig, self.ax = plt.subplots(figsize=(width / 10, height / 10), dpi=200)
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.axis("off")
        self.font = FONT[lang]

    def box(self, x: float, y: float, w: float, h: float, text: str, kind: str, size: float = 7.5,
            bold: bool = False) -> tuple[float, float, float, float]:
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                         fc=COLOR[kind], ec="#333333", lw=0.8))
        self.ax.text(x + w / 2, y + h / 2, text.replace("$", r"\$"), ha="center", va="center", fontsize=size, family=self.font,
                     weight="bold" if bold else "normal", linespacing=1.25)
        return x, y, w, h

    def container(self, x: float, y: float, w: float, h: float, label: str = "", dotted: bool = True,
                  fill: str = "none", label_at: str = "tl") -> None:
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2.5", fc=fill,
                                         ec="#222222", lw=1.0, ls=(0, (1, 1.6)) if dotted else "-"))
        if label:
            if label_at == "br":
                self.ax.text(x + w - 1.5, y + 1.6, label, ha="right", va="bottom", fontsize=9, family=self.font)
            else:
                self.ax.text(x + 2, y + h - 2, label, ha="left", va="top", fontsize=9, family=self.font, weight="bold")

    def plus(self, x: float, y: float) -> tuple[float, float]:
        self.ax.add_patch(Circle((x, y), 1.4, fc="white", ec="#333333", lw=0.8))
        self.ax.text(x, y, "+", ha="center", va="center", fontsize=9)
        return x, y

    def arrow(self, points: list[tuple[float, float]], dashed: bool = False) -> None:
        """A line through the points with an arrow head at the last one."""
        for a, b in zip(points[:-2], points[1:-1], strict=False):
            self.ax.plot([a[0], b[0]], [a[1], b[1]], color="#222222", lw=0.9, ls="--" if dashed else "-")
        self.ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=8,
                                          color="#222222", lw=0.9, ls="--" if dashed else "-",
                                          shrinkA=0, shrinkB=0))

    def text(self, x: float, y: float, s: str, size: float = 8, ha: str = "left", italic: bool = False) -> None:
        self.ax.text(x, y, s.replace("$", r"\$"), ha=ha, va="center", fontsize=size, family=self.font,
                     style="italic" if italic else "normal")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(self.fig)


def block(c: Canvas, x0: float, y: float, n_layers: int, attn: str, ffn: str, t: dict[str, str],
          start: float | None = None) -> float:
    """One pre-norm decoder layer, repeated n_layers times: RMSNorm → attention → + → RMSNorm → SwiGLU → +.
    Returns the x where the residual stream leaves the block."""
    h = 10
    c.container(x0, y - 9.5, 101, 21, dotted=False, fill="#f2f2f2")
    c.text(x0 - 0.5, y + 13, f"{n_layers} ×", size=10)
    n1 = c.box(x0 + 5, y - h / 2, 11, h, t["norm"], "norm")
    at = c.box(x0 + 20, y - h / 2, 26, h, attn, "attn", size=7)
    p1 = c.plus(x0 + 50, y)
    n2 = c.box(x0 + 54, y - h / 2, 11, h, t["norm"], "norm")
    ff = c.box(x0 + 69, y - h / 2, 22, h, ffn, "ffn", size=7)
    p2 = c.plus(x0 + 95, y)
    c.arrow([(x0 - 6 if start is None else start, y), (n1[0], y)])     # from the embedding into the first layer
    c.arrow([(n1[0] + n1[2], y), (at[0], y)])
    c.arrow([(at[0] + at[2], y), (p1[0] - 1.4, y)])
    c.arrow([(p1[0] + 1.4, y), (n2[0], y)])
    c.arrow([(n2[0] + n2[2], y), (ff[0], y)])
    c.arrow([(ff[0] + ff[2], y), (p2[0] - 1.4, y)])
    top = y + 8                                      # the residual stream skips each sub-layer
    c.arrow([(x0 + 2.5, y), (x0 + 2.5, top), (p1[0], top), (p1[0], y + 1.4)])
    c.arrow([(x0 + 52, y), (x0 + 52, top), (p2[0], top), (p2[0], y + 1.4)])
    c.text(x0 + 26, top + 1.3, t["residual"], size=6.5, italic=True)
    return p2[0] + 1.4


TEXT = {
    "en": {"norm": "RMSNorm", "residual": "residual stream (pre-norm)"},
    "zh": {"norm": "RMSNorm", "residual": "殘差連接(先正規化)"},
}


def buyer_gpt(lang: str) -> None:
    import torch
    cfg = torch.load(ML / "runs/sft_b2/best.pt", map_location="cpu", weights_only=False)["config"]
    d, L, H, KV, V, T = cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg["n_kv_heads"], cfg["vocab_size"], \
        cfg["max_seq_len"]
    ffn = int(-(-(8 * d / 3) // 64) * 64)                 # GPTConfig.ffn_dim: 8/3 · d, rounded up to 64
    en = lang == "en"
    t = TEXT[lang]
    c = Canvas(176, 62, lang)
    c.container(1, 30, 174, 30, "Buyer GPT B2" if en else "出價 GPT(B2)")
    c.box(4, 38, 24, 14, "Prompt\n<|item|> cat\n<|persona|> card\n<|log|> story\n+ task token" if en else
          "提示\n<|item|> 貓\n<|persona|> 買家卡片\n<|log|> 前文\n+ 任務 token", "io", size=6.5, bold=False)
    c.box(32, 40, 16, 10, f"Tokenizer\nSentencePiece\n{V:,} tokens" if en else f"分詞器\nSentencePiece\n{V:,} 個 token",
          "pos", size=6.5)
    c.box(52, 40, 15, 10, f"Token\nembedding\n{V:,} × {d}" if en else f"token\n嵌入\n{V:,} × {d}", "embed", size=6.5)
    c.arrow([(28, 45), (32, 45)])
    c.arrow([(48, 45), (52, 45)])
    attn = (f"Masked grouped-query\nself-attention\n{H} query / {KV} key-value heads\n+ RoPE positions" if en else
            f"遮罩的分組查詢\n自注意力\n{H} 個 query 頭 / {KV} 個 KV 頭\n+ RoPE 位置")
    ffn_t = f"SwiGLU\nfeed-forward\n{d} → {ffn} → {d}" if en else f"SwiGLU\n前饋層\n{d} → {ffn} → {d}"
    end = block(c, 73, 45, L, attn, ffn_t, t)
    c.container(1, 1, 174, 26, "Output" if en else "輸出")
    n = c.box(118, 8, 13, 10, "Final\nRMSNorm" if en else "最後的\nRMSNorm", "norm", size=6.5)
    o = c.box(96, 8, 18, 10, f"Linear {d} → {V:,}\n(tied to the\nembedding)" if en else f"線性層 {d} → {V:,}\n(與嵌入\n共用權重)",
              "out", size=6.5)
    s = c.box(74, 8, 18, 10, "softmax, sample\n(temperature 0.7,\ntop-k 50)" if en else "softmax 抽樣\n(溫度 0.7、\ntop-k 50)",
              "attn", size=6.5)
    r = c.box(38, 6, 32, 14, "One line, e.g. a reaction:\n+ | A cat that stays in suits\nmy busy road. Up to $80. | $80"
              if en else "一句話,例如反應:\n+ | A cat that stays in suits\nmy busy road. Up to $80. | $80", "io", size=6.5)
    g = c.box(4, 6, 30, 14, "Game reads it: move,\nwhat the buyer says, new offer\n(open / ask / react / imagine)"
              if en else "遊戲讀出:方向、\n買家說的話、新出價\n(開場/提問/反應/想像)", "data", size=6.5)
    c.arrow([(end, 45), (172, 45), (172, 13), (n[0] + n[2], 13)])
    c.arrow([(n[0], 13), (o[0] + o[2], 13)])
    c.arrow([(o[0], 13), (s[0] + s[2], 13)])
    c.arrow([(s[0], 13), (r[0] + r[2], 13)])
    c.arrow([(r[0], 13), (g[0] + g[2], 13)])
    c.arrow([(83, 18), (83, 23), (20, 23), (20, 38)], dashed=True)
    c.text(22, 24.6, "each new token is added to the input until <|end|> (KV cache)" if en else
           "每個新 token 接回輸入,直到 <|end|>(KV 快取)", size=6.5, italic=True)
    params = 71.0
    c.text(24, 33, (f"{params} M parameters · context {T:,} tokens · pre-trained on 1.28 B FineWeb-Edu tokens, then "
                   f"fine-tuned on 223 cat sales (loss on the buyer's line only)") if en else
           (f"{params:.1f}M 參數 · 上下文 {T:,} 個 token · 先用 12.8 億個 FineWeb-Edu token 預訓練,"
            f"再用 223 場賣貓對話做 SFT(只對買家那句話算 loss)"), size=6.5)
    c.save(OUT / f"buyer_gpt_{lang}.png")


def brush_gpt(lang: str) -> None:
    run = json.loads((ML / "runs/brushgpt/I_aligned_points_eyes/config.json").read_text())
    m, cfg = run["model"], run["config"]
    d, L, H, V, T = m["d_model"], m["n_layers"], m["n_heads"], m["vocab_size"], m["max_seq_len"]
    ffn = int(-(-(8 * d / 3) // 64) * 64)
    en = lang == "en"
    t = TEXT[lang]
    c = Canvas(176, 62, lang)
    c.container(1, 30, 174, 30, "Brush GPT (run I)" if en else "油畫 transformer(I 版)")
    c.box(4, 37, 24, 16, "Painting as tokens\nBOS · 9 face points (x, y)\nSTAGE · stroke: x y angle\nsize paint …"
          "\n4 stages: 24/64/160/360" if en else "一幅畫寫成 token\nBOS · 9 個臉部點 (x, y)\nSTAGE · 每筆:x y 角度\n大小 顏色 …"
          "\n4 階段:24/64/160/360 筆", "io", size=6)
    e = c.box(33, 40, 15, 10, f"Token\nembedding\n{V:,} × {d}" if en else f"token\n嵌入\n{V:,} × {d}", "embed", size=6.5)
    p = c.plus(53, 45)
    c.box(46, 31.5, 14, 8, f"Learned position\nembedding\n{T:,} × {d}" if en else f"學習的\n位置嵌入\n{T:,} × {d}", "pos",
          size=6)
    c.arrow([(28, 45), (e[0], 45)])
    c.arrow([(e[0] + e[2], 45), (p[0] - 1.4, 45)])
    c.arrow([(53, 39.5), (53, 43.6)])
    attn = (f"Masked multi-head\nself-attention\n{H} heads, dropout {cfg['dropout']}\n+ RoPE positions" if en else
            f"遮罩的多頭\n自注意力\n{H} 個頭、dropout {cfg['dropout']}\n+ RoPE 位置")
    ffn_t = f"SwiGLU\nfeed-forward\n{d} → {ffn} → {d}" if en else f"SwiGLU\n前饋層\n{d} → {ffn} → {d}"
    end = block(c, 69, 45, L, attn, ffn_t, t, start=p[0] + 1.4)
    c.container(1, 1, 174, 26, "Output" if en else "輸出")
    n = c.box(126, 8, 13, 10, "Final\nRMSNorm" if en else "最後的\nRMSNorm", "norm", size=6.5)
    o = c.box(98, 7, 24, 12, "Tied output, only the\nrows of this position's\nfield (x / y / angle /\nsize / paint)" if en
              else "共用權重的輸出層,\n只用這個位置那一欄\n的 token(x/y/角度/\n大小/顏色)", "out", size=6)
    s = c.box(76, 8, 18, 10, "softmax, sample\nnext token" if en else "softmax 抽樣\n下一個 token", "attn", size=6.5)
    r = c.box(44, 6, 28, 14, "Renderer: 608 straight\nbrush strokes, big to small,\non a 256 × 256 canvas" if en
              else "渲染器:608 筆直線筆觸,\n由大到小,畫在\n256 × 256 的畫布上", "data", size=6.5)
    g = c.box(4, 6, 36, 14, "Oil-painting cat face\n(trained on 3,413 AFHQ cats aligned by\nthe eyes; eye strokes count "
              f"×{cfg['eye_weight']:g} in the loss)" if en else
              f"油畫風格的貓臉\n(用 3,413 隻依眼睛對齊的 AFHQ 貓訓練;\n眼睛附近的筆觸 loss ×{cfg['eye_weight']:g})", "io", size=6.5)
    c.arrow([(end, 45), (172, 45), (172, 13), (n[0] + n[2], 13)])
    c.arrow([(n[0], 13), (o[0] + o[2], 13)])
    c.arrow([(o[0], 13), (s[0] + s[2], 13)])
    c.arrow([(s[0], 13), (r[0] + r[2], 13)])
    c.arrow([(r[0], 13), (g[0] + g[2], 13)])
    c.arrow([(85, 18), (85, 23), (16, 23), (16, 37)], dashed=True)
    c.text(18, 24.6, f"each new token is added to the input, {T:,} tokens in all (KV cache)" if en else
           f"每個新 token 接回輸入,總共 {T:,} 個 token(KV 快取)", size=6.5, italic=True)
    c.text(20, 33, f"{run['parameters'] / 1e6:.1f} M parameters · one A100 on Modal" if en else
           f"{run['parameters'] / 1e6:.1f}M 參數 · Modal 上一張 A100 訓練", size=6.5)
    c.save(OUT / f"brush_gpt_{lang}.png")


if __name__ == "__main__":
    for lang in ("en", "zh"):
        buyer_gpt(lang)
        brush_gpt(lang)
    print(sorted(p.name for p in OUT.glob("*.png")))
