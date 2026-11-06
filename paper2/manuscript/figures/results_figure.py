"""Figure 2 of paper 2: the Learning condition's goals in blocks of five episodes for each run, with the points
where a heading or report procedure was adopted. Runs A to D are scored streams 4301 to 4304, in that order.

Claude, 3 November 2026. Reads only paper2/reproduction/results-tables-claude-20261103.json, which holds every plotted
value (per-episode labels and the result of each learning attempt). Writes results-goals-by-block.svg for the HTML
copy, .pdf for the LaTeX project and .png for the Word copy.

    uv run --no-project --with matplotlib python paper2/manuscript/figures/results_figure.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLES = HERE.parents[1] / "reproduction/results-tables-claude-20261103.json"
RUNS = {"4301": "Run A", "4302": "Run B", "4303": "Run C", "4304": "Run D"}
PHASES = ("Still air", "Wind", "Orbit, then delay", "Mixed", "Help cases")
NAME = "results-goals-by-block"
# The first two slots of the validated reference palette (dataviz skill), with ink, muted text and rule colours.
INK, MUTED, RULE, BLUE, ORANGE, SURFACE = "#0b0b0b", "#52514e", "#c9c8c3", "#2a78d6", "#eb6834", "#ffffff"


def blocks(episodes):
    """Goals met without and with help in each block of five episodes, in run order."""
    alone, helped = [0] * 20, [0] * 20
    for i, e in enumerate(episodes):
        if e["label"] == "met_alone":
            alone[i // 5] += 1
        elif e["label"] == "met_with_help":
            helped[i // 5] += 1
    return alone, helped


def adoptions(attempts):
    """For each boundary, numbered by the episode it follows, the decisions whose candidate was adopted there."""
    out = {}
    for a in attempts:
        phase, episode = a["boundary"].split("-p")[1].split("-e")
        after = (int(phase) - 1) * 20 + int(episode) + 1
        for decision, passed in a["results"]:
            if passed:
                out.setdefault(after, []).append(decision)
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
        "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none",
        "pdf.fonttype": 42, "savefig.dpi": 300})

    data = json.loads(TABLES.read_text())["per_stream"]
    streams = list(data)
    fig, axes = plt.subplots(len(streams), 1, figsize=(6.3, 5.4), sharex=True)
    x = list(range(1, 21))
    for row, (ax, s) in enumerate(zip(axes, streams)):
        alone, helped = blocks(data[s]["L_episodes"])
        ax.bar(x, alone, width=0.72, color=BLUE, edgecolor=SURFACE, linewidth=0.8, zorder=2)
        ax.bar(x, helped, width=0.72, bottom=alone, color=ORANGE, edgecolor=SURFACE, linewidth=0.8, zorder=2)
        for edge in (4.5, 8.5, 12.5, 16.5):
            ax.axvline(edge, color=RULE, linewidth=0.6, zorder=1)
        # Adoption markers sit just above the bars, at the boundary after the episode they follow.
        for after, decisions in adoptions(data[s]["learning_attempts"]).items():
            bx = after / 5 + 0.5
            for k, decision in enumerate(sorted(decisions)):
                ax.plot(bx, 5.75 + 0.75 * k, marker="^" if decision == "heading" else "o", markersize=4.2,
                        color=INK, markeredgecolor=SURFACE, markeredgewidth=0.5, linestyle="none", zorder=3,
                        clip_on=False)
        ax.set_ylim(0, 5)
        ax.set_yticks([0, 5])
        ax.set_xlim(0.4, 20.6)
        ax.set_ylabel(RUNS[s], rotation=0, ha="right", va="center", labelpad=8)
        if row == 0:
            for name, mid in zip(PHASES, (2.5, 6.5, 10.5, 14.5, 18.5)):
                ax.text(mid, 7.6, name, ha="center", va="bottom", color=MUTED, fontsize=7)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([f"{5 * i - 4}–{5 * i}" if i in (1, 5, 9, 13, 17, 20) else "" for i in x])
    axes[-1].set_xlabel("Episodes, in blocks of five")
    fig.supylabel("Goals met in the block (of 5)", x=0.0, fontsize=8, color=INK)
    handles = [Patch(color=BLUE, label="Met without a help request"), Patch(color=ORANGE, label="Met with a help request"),
               Line2D([], [], marker="^", color=INK, linestyle="none", markersize=4.5,
                      label="Heading procedure adopted"),
               Line2D([], [], marker="o", color=INK, linestyle="none", markersize=4.5,
                      label="Report procedure adopted")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.53, 1.07),
               handlelength=1.2, columnspacing=1.4)
    fig.subplots_adjust(hspace=0.62, top=0.88, left=0.12)
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(HERE / f"{NAME}.{suffix}", bbox_inches="tight", pad_inches=0.03, facecolor=SURFACE,
                    metadata={"Date": None} if suffix in ("svg", "pdf") else None)


if __name__ == "__main__":
    main()
