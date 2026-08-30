"""Annotation-tool figure: how anyone annotates a new spaceborne video with it.

A schematic of the pipeline the tool offers -- load, annotate, refine, verify,
export -- with the affordance behind each step named as it is named in the UI.
The subject is the tool as a reusable instrument, not the curation this paper
happened to run through it. Companion to ``plot_annotation_tool.py``, which
shows the surfaces themselves.

No data is read: this figure is a diagram, so it is reproducible anywhere.
Line breaks are written by hand rather than wrapped, because every box has a
fixed width in inches and a wrapped line that overflows is invisible until the
PDF is compiled.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import apply_neurips_style  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "wacv-2027-author-kit-template" / "plots"

INK = "#1a1a1a"
EDGE = "#8a8a8a"
FILL = "#f3f3f1"
FILL_ACCENT = "#e6edf5"
FILL_SUB = "#ffffff"
ARROW = "#4f4f4f"

W = 7.0  # inches: \textwidth in the WACV two-column layout


def panel(ax, x, y, w, h, title, body, fc=FILL, title_size=8.4, body_size=7.0,
          pad=0.075, gap=0.16):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=0.045",
                                fc=fc, ec=EDGE, lw=0.7, mutation_aspect=1.0))
    ax.text(x + pad, y + h - pad, title, ha="left", va="top", fontsize=title_size,
            fontweight="bold", color=INK)
    if body:
        ax.text(x + pad, y + h - pad - gap, body, ha="left", va="top",
                fontsize=body_size, color=INK, linespacing=1.45)


def arrow(ax, tail, head, rad=0.0):
    ax.add_patch(FancyArrowPatch(tail, head, arrowstyle="-|>", mutation_scale=8,
                                 lw=0.8, color=ARROW,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1.0, shrinkB=1.0))


LOAD = (
    "Point the tool at any video\n"
    "sequence: image frames or\n"
    "a video file, plus the\n"
    "categories it may contain.\n"
    "\n"
    "An existing annotation, if\n"
    "there is one, loads as a\n"
    "layer to scrutinise rather\n"
    "than to trust; with none,\n"
    "the sequence starts empty\n"
    "and every track in it is\n"
    "drawn here.\n"
    "\n"
    "Nothing is specific to the\n"
    "sources of this benchmark,\n"
    "or to overhead imagery: any\n"
    "video whose cost is its\n"
    "many similar objects fits\n"
    "this loop."
)

ANNOTATE = (
    "New object here → prompt one frame → Segment → Propagate → Save as track: one anchor\n"
    "frame yields a whole track. Prompting is SAM's own model — positive clicks, negative\n"
    "clicks and a two-corner box, in any combination, refined by adding to the prompt. One\n"
    "click suffices on open ground; on a 5 px car it returns the road, so there a box is used."
)

REFINE = [
    # (title, body, width factor). The exemplar sweep is the widest because it
    # is the point of the figure: one annotated object, every other instance.
    ("Find all like this",
     "One annotated object becomes a visual\n"
     "exemplar: SAM 3 pools boxes sampled\n"
     "along its track into a prompt token and\n"
     "its grounding head finds that concept\n"
     "on every other frame — annotate one\n"
     "aircraft, get the aircraft.\n"
     "\n"
     "Background boxes go in as negatives,\n"
     "and candidates far from the exemplar's\n"
     "area are dropped.", 1.45),
    ("Correct",
     "Proposals arrive as a wall\n"
     "of scored tiles: strike out\n"
     "the wrong ones, or reject\n"
     "everything below a score.\n"
     "Nothing is written until\n"
     "you adopt them.\n"
     "\n"
     "Two corners fix a box, or\n"
     "S snaps it to a SAM 3 mask,\n"
     "on that frame alone.", 1.00),
    ("Clean up tracks",
     "Merge duplicates · delete\n"
     "static tracks · drop\n"
     "oversized boxes · drop\n"
     "switched stretches · trim\n"
     "the parked start · relabel\n"
     "a mis-classed track.\n"
     "\n"
     "Identity is centre distance\n"
     "in object widths, not IoU.", 1.05),
]

VERIFY = (
    "Scan the object grid, check the whole frame, then play the sequence back. Accepting is refused\n"
    "until it has been played, because a box that drifts over 200 frames is invisible in every still\n"
    "view. Any later edit clears the sign-off; flagging parks a decision, excluding needs a reason."
)

EXPORT = (
    "The 11-column MOT rows\n"
    "per sequence, plus a\n"
    "manifest recording what\n"
    "was accepted, flagged or\n"
    "excluded, and why.\n"
    "\n"
    "Every decision lives in\n"
    "one JSON file, and the\n"
    "video's own files are\n"
    "never written.\n"
    "\n"
    "A sequence annotated\n"
    "here is interchangeable\n"
    "with the released ones."
)


def main():
    apply_neurips_style(base_size=9.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    m, gap, vgap = 0.05, 0.17, 0.16
    w_side = 1.24
    x_mid = m + w_side + gap
    w_mid = W - 2 * m - 2 * (w_side + gap)
    x_right = x_mid + w_mid + gap

    h_b, h_d = 0.86, 0.62
    sub_h = 1.32
    h_c = sub_h + 0.36
    y_d = 0.06
    y_c = y_d + h_d + vgap
    y_b = y_c + h_c + vgap
    top = y_b + h_b
    H = top + 0.06

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_axis_off()

    panel(ax, m, y_d, w_side, top - y_d, "1  Load a video", LOAD, fc=FILL_ACCENT)
    panel(ax, x_mid, y_b, w_mid, h_b,
          "2  Annotate one object", ANNOTATE)
    panel(ax, x_mid, y_c, w_mid, h_c,
          "3  Scale it up with one exemplar, then fix what it got wrong", "")
    panel(ax, x_mid, y_d, w_mid, h_d, "4  Verify by watching it", VERIFY)
    panel(ax, x_right, y_d, w_side, top - y_d, "5  Export", EXPORT, fc=FILL_ACCENT)

    sub_gap = 0.055
    unit = (w_mid - 4 * sub_gap) / sum(f for *_, f in REFINE)
    x = x_mid + sub_gap
    for t, b, f in REFINE:
        panel(ax, x, y_c + 0.06, unit * f, sub_h, t, b, fc=FILL_SUB,
              title_size=7.4, body_size=6.4, pad=0.06, gap=0.145)
        x += unit * f + sub_gap

    # Flow. The two side columns span the full height, so they are entered and
    # left at their vertical middle.
    mid_y = (y_d + top) / 2
    arrow(ax, (m + w_side + 0.015, mid_y), (x_mid - 0.015, y_b + h_b / 2), rad=0.20)
    arrow(ax, (x_mid + w_mid * 0.5, y_b - 0.012), (x_mid + w_mid * 0.5, y_c + h_c + 0.012))
    arrow(ax, (x_mid + w_mid * 0.72, y_c - 0.012),
          (x_mid + w_mid * 0.72, y_d + h_d + 0.012))
    arrow(ax, (x_mid + w_mid * 0.28, y_d + h_d + 0.012),
          (x_mid + w_mid * 0.28, y_c - 0.012))
    ax.text(x_mid + w_mid * 0.28 - 0.045, (y_c + y_d + h_d) / 2, "go back",
            ha="right", va="center", fontsize=6.3, color=ARROW)
    arrow(ax, (x_mid + w_mid + 0.015, y_d + h_d / 2), (x_right - 0.015, mid_y), rad=0.20)

    out = OUT_DIR / "annotation_workflow.pdf"
    fig.savefig(out, bbox_inches=None)
    fig.savefig(OUT_DIR / "annotation_workflow.png", dpi=240, bbox_inches=None)
    plt.close(fig)
    print(f"saved: {out}  ({W} x {H:.2f} in)")


if __name__ == "__main__":
    main()
