"""Filesystem roots, read from the environment.

No installation path is hard-coded in this repository. Point the roots at your
own copies before running anything::

    export DATA_ROOT=/path/to/data          # source datasets and the built release
    export EXPERIMENT_ROOT=/path/to/runs    # where training / evaluation writes

``DATA_ROOT`` is expected to contain:

* ``data/<dataset>/`` — the source datasets, as downloaded from their authors;
* ``release/space_tracker/`` — the built Space-Tracker release;
* ``experiments/Detection/…`` — the cached detections the trackers consume.

Configs spell the same roots as ``${DATA_ROOT}`` / ``${EXPERIMENT_ROOT}``, and
:func:`load_config` expands them at load time, so no config needs editing to
move between machines.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

#: This checkout. Configs spell it ``${REPO_ROOT}`` so a vendored tracker or a
#: sibling config can be referenced without an installation-specific path.
REPO_ROOT = str(Path(__file__).resolve().parent)

#: Source datasets, the built release, and the cached detections.
DATA_ROOT = os.environ.get("DATA_ROOT", "/data/anon")

#: Where training and evaluation runs write their outputs.
EXPERIMENT_ROOT = os.environ.get("EXPERIMENT_ROOT", "/work/anon/experiments")

_ROOTS = {
    "${DATA_ROOT}": DATA_ROOT,
    "${EXPERIMENT_ROOT}": EXPERIMENT_ROOT,
    "${REPO_ROOT}": REPO_ROOT,
}


def expand(text: str) -> str:
    """Expand ``${DATA_ROOT}`` / ``${EXPERIMENT_ROOT}`` / ``${REPO_ROOT}``.

    Those three resolve through this module, so they fall back to the values
    above when unset rather than to an empty string. Any other ``$VAR`` is left
    to :func:`os.path.expandvars`.
    """
    for token, value in _ROOTS.items():
        text = text.replace(token, value)
    return os.path.expandvars(text)


def load_config(source: Any) -> Any:
    """``yaml.safe_load`` with the roots expanded.

    Accepts an open file object, or a path to one.
    """
    text = source.read() if hasattr(source, "read") else Path(source).read_text()
    return yaml.safe_load(expand(text))
