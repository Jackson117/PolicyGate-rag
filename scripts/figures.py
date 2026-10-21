"""The README's figures, from results/ alone (no runs, no labels, no network).

    uv run python scripts/figures.py          # writes docs/figures/*.svg

Static SVGs for a page that cannot hover; the interactive versions live in
policygate's dashboard. One y-axis per chart, intervals drawn, identity never
carried by colour alone. Colours are the dataviz reference palette (light mode,
the surface GitHub renders on by default); slot 1 is the measure a chart is
about, slot 2 its comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "figures"

SURFACE, INK, INK2, MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")
SERIES1, SERIES2 = "#2a78d6", "#eb6834"

ARMS = ("A0", "A1", "A2", "A3", "A5", "A4")
SHORT = {"A0": "A0 floor: no knowledge", "A1": "A1 tau2 BM25 tool", "A2": "A2 naive RAG",
         "A3": "A3 agentic RAG + Jev", "A5": "A5 A3 + working set + products",
         "A4": "A4 ceiling: golden documents"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "text.color": INK2,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titlelocation": "left", "axes.titleweight": "bold",
    "xtick.color": MUTED, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "svg.fonttype": "none",
})


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def _frame(ax, xlabel: str, xmax: float = 1.0) -> None:
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)


def interval_rows(ax, labels, intervals, *, color, hollow=(), offset=0.0, label=None,
                  value_labels=True):
    """One row per label: a whisker for the interval, a dot for the estimate."""
    for i, (name, iv) in enumerate(zip(labels, intervals, strict=True)):
        if iv is None:
            continue
        y = i + offset
        ax.plot([iv["low"], iv["high"]], [y, y], color=color, lw=2, solid_capstyle="round")
        open_ = name in hollow
        ax.plot(iv["estimate"], y, "o", ms=8, color=color,
                mfc=SURFACE if open_ else color, mew=2,
                label=label if i == 0 else None)
        if value_labels:
            ax.annotate(f"{iv['estimate']:.3f}", (iv["high"], y), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(labels)))
    if not ax.yaxis_inverted():
        ax.invert_yaxis()


def pass1_by_arm(arms: dict) -> plt.Figure:
    present = [a for a in ARMS if "pass_k" in arms.get(a, {})]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    interval_rows(ax, present, [arms[a]["pass_k"]["pass^1"] for a in present], color=SERIES1,
                  hollow={a for a in present if arms[a]["status"] == "partial"})
    ax.set_yticklabels([SHORT[a] + (" (partial)" if arms[a]["status"] == "partial" else "")
                        for a in present])
    _frame(ax, "pass^1, 95% bootstrap interval over 30 tasks", xmax=0.45)
    ax.set_title("Task success by arm")
    fig.tight_layout()
    return fig


def coverage_by_arm(arms: dict) -> plt.Figure:
    present = [a for a in ARMS if "action_coverage" in (arms.get(a, {}).get("retrieval") or {})]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    interval_rows(ax, present, [arms[a]["retrieval"]["coverage"] for a in present],
                  color=SERIES2, offset=-0.14, label="any time in the episode")
    interval_rows(ax, present, [arms[a]["retrieval"]["action_coverage"] for a in present],
                  color=SERIES1, offset=0.14, label="at each proposed tool call")
    ax.set_yticklabels([SHORT[a] + (" (partial)" if arms[a]["status"] == "partial" else "")
                        for a in present])
    _frame(ax, "share of the task's required documents in the model's context")
    ax.set_title("Required documents the model had", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2, frameon=False,
              borderaxespad=0, handletextpad=0.3)
    fig.tight_layout()
    return fig


def replay(report: dict) -> plt.Figure:
    names = list(report["policies"])
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    interval_rows(ax, names, [report["policies"][n]["mean_action_coverage"] for n in names],
                  color=SERIES1)
    ax.set_yticklabels([n.replace("ws", "keep ").replace(" kept", "").replace(" pool", ", pool")
                        .replace("unbounded", "keep all") for n in names])
    _frame(ax, "required documents in context when a tool call is proposed (replay)", 0.65)
    ax.set_title("Replay of A3's conversations under other context policies")
    fig.tight_layout()
    return fig


def judges(stage1: dict) -> plt.Figure:
    common = stage1["common"]
    names = sorted(common["judges"], key=lambda j: -common["judges"][j]["mrr"]["estimate"])
    labels = [stage1["judges"][j]["label"] for j in names]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6), sharey=True)
    for ax, (key, title) in zip(axes, [("mrr", "MRR"), ("coverage_at_k", "Coverage@5")],
                                strict=True):
        interval_rows(ax, names, [common["judges"][j][key] for j in names], color=SERIES1)
        ax.set_yticklabels(labels)
        _frame(ax, title)
    fig.suptitle(f"Stage 1: re-ranking judges on {common['queries']} queries from "
                 f"{common['tasks']} tasks", x=0.01, ha="left", fontsize=12,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def cost_vs_success(arms: dict) -> plt.Figure:
    present = [a for a in ARMS if "pass_k" in arms.get(a, {})]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for a in present:
        e = arms[a]
        x = e["cost"]["usd_mean"]
        iv = e["pass_k"]["pass^1"]
        ax.plot([x, x], [iv["low"], iv["high"]], color=SERIES1, lw=2, alpha=0.35)
        ax.plot(x, iv["estimate"], "o", ms=8, color=SERIES1,
                mfc=SURFACE if e["status"] == "partial" else SERIES1, mew=2)
        ax.annotate(a, (x, iv["estimate"]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK2)
    ax.set_xlim(0, 0.08)
    ax.set_ylim(0, 0.42)
    ax.set_xlabel("USD per episode, agent plus user simulator (judge billed separately)")
    ax.set_ylabel("pass^1")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Cost and success per arm")
    fig.tight_layout()
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arms = load("stage2_arms.json")["arms"]
    figures = {
        "pass1_by_arm.svg": pass1_by_arm(arms),
        "coverage_by_arm.svg": coverage_by_arm(arms),
        "replay_a3.svg": replay(load("replay_a3.json")),
        "judges.svg": judges(load("stage1_judges.json")),
        "cost_vs_success.svg": cost_vs_success(arms),
    }
    for name, fig in figures.items():
        fig.savefig(OUT / name)
        plt.close(fig)
        print(OUT / name)


if __name__ == "__main__":
    main()
