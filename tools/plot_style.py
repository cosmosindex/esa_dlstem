"""Shared matplotlib style for paper figures.

NeurIPS uses Times-style serif fonts (LaTeX `times` package = Times Roman).
We pick the best available Times-equivalent on the current system and fall
back gracefully. Math is rendered with STIX so glyphs match the body text.

Usage:
    from plot_style import apply_neurips_style
    apply_neurips_style()
    # ... build your figure ...
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.font_manager as fm

# Ordered preference: real Times first, then Times-metric clones, then STIX
# (Times-clone shipped with matplotlib), then platform serifs.
_SERIF_PREFERENCE = [
    "Times New Roman",
    "Times",
    "Nimbus Roman",
    "Liberation Serif",
    "TeX Gyre Termes",
    "STIX Two Text",
    "STIXGeneral",
    "DejaVu Serif",
]


def _pick_serif() -> list[str]:
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = [name for name in _SERIF_PREFERENCE if name in available]
    # Always end with the matplotlib default so we never crash if nothing matches.
    if "DejaVu Serif" not in chosen:
        chosen.append("DejaVu Serif")
    return chosen


def apply_neurips_style(base_size: float = 9.0) -> None:
    """Apply Times-style serif fonts and tight defaults suited to NeurIPS figures.

    base_size: matches a typical NeurIPS body-text size (9-10pt).
    """
    serif_chain = _pick_serif()
    mpl.rcParams.update({
        # Fonts -- body
        "font.family": "serif",
        "font.serif": serif_chain,
        "font.size": base_size,
        # Fonts -- math (STIX is Times-compatible; cm is the LaTeX default)
        "mathtext.fontset": "stix",
        # Per-element sizes (slight contrast around base_size)
        "axes.titlesize": base_size + 1,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "legend.fontsize": base_size - 1.5,
        "figure.titlesize": base_size + 2,
        # PDF/PS: embed Type 42 (TrueType) so submission systems don't reject Type 3.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Line / axes aesthetics
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.25,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


__all__ = ["apply_neurips_style"]
